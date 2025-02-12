import argparse
import numpy as np
from datasets import load_dataset
from transformers import (
    AutoModelForQuestionAnswering,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    default_data_collator,
)
import evaluate
import smtplib
from email.mime.text import MIMEText
import datetime
import json
import os
import logging
import traceback
from typing import Dict, List
import torch
import gc
from time import sleep

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.FileHandler("experiments.log"), logging.StreamHandler()],
)


class ExperimentManager:
    def __init__(self, email: str):
        self.email = email
        self.results_dir = "experiment_results"
        os.makedirs(self.results_dir, exist_ok=True)
        self.failed_experiments = []
        self.successful_experiments = []

        # Load experiment configurations
        self.load_configurations()

    def load_configurations(self):
        # Load models and datasets from JSON files
        with open("models.json", "r") as f:
            self.models = json.load(f)["models"]
        with open("datasets.json", "r") as f:
            self.datasets = json.load(f)["datasets"]

    def cleanup_gpu(self):
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.memory.empty_cache()
        sleep(5)  # Give system time to free resources

    def send_email(self, subject: str, content: str):
        smtp_server = "smtp.gmail.com"
        smtp_port = 587
        sender_email = "nghiemptce160353@fpt.edu.vn"
        sender_password = "tfjdtvwrdlrqqlzy"

        msg = MIMEText(content)
        msg["Subject"] = subject
        msg["From"] = sender_email
        msg["To"] = self.email

        try:
            server = smtplib.SMTP(smtp_server, smtp_port)
            server.starttls()
            server.login(sender_email, sender_password)
            server.send_message(msg)
            server.quit()
            logging.info("Email sent successfully")
        except Exception as e:
            logging.error(f"Failed to send email: {str(e)}")

    def preprocess_squad(self, examples, tokenizer, max_length=384, stride=128):
        try:
            questions = [q.strip() for q in examples["question"]]
            contexts = [c.strip() for c in examples["context"]]

            tokenized = tokenizer(
                questions,
                contexts,
                max_length=max_length,
                stride=stride,
                truncation="only_second",
                return_overflowing_tokens=True,
                return_offsets_mapping=True,
                padding="max_length",
            )

            sample_mapping = tokenized.pop("overflow_to_sample_mapping")
            offset_mapping = tokenized.pop("offset_mapping")

            tokenized["start_positions"] = []
            tokenized["end_positions"] = []

            for i, offsets in enumerate(offset_mapping):
                input_ids = tokenized["input_ids"][i]
                cls_index = input_ids.index(tokenizer.cls_token_id)
                sequence_ids = tokenized.sequence_ids(i)

                sample_idx = sample_mapping[i]
                answers = examples["answers"][sample_idx]

                if len(answers["answer_start"]) == 0:
                    tokenized["start_positions"].append(cls_index)
                    tokenized["end_positions"].append(cls_index)
                else:
                    start_char = answers["answer_start"][0]
                    end_char = start_char + len(answers["text"][0])

                    token_start_index = 0
                    while sequence_ids[token_start_index] != 1:
                        token_start_index += 1

                    token_end_index = len(input_ids) - 1
                    while sequence_ids[token_end_index] != 1:
                        token_end_index -= 1

                    if not (
                        offsets[token_start_index][0] <= start_char
                        and offsets[token_end_index][1] >= end_char
                    ):
                        tokenized["start_positions"].append(cls_index)
                        tokenized["end_positions"].append(cls_index)
                    else:
                        while (
                            token_start_index < len(offsets)
                            and offsets[token_start_index][0] <= start_char
                        ):
                            token_start_index += 1
                        tokenized["start_positions"].append(token_start_index - 1)

                        while offsets[token_end_index][1] >= end_char:
                            token_end_index -= 1
                        tokenized["end_positions"].append(token_end_index + 1)

            return tokenized
        except Exception as e:
            logging.error(f"Error in preprocessing: {str(e)}")
            raise

    def compute_metrics(self, eval_preds):
        try:
            metric = evaluate.load("squad")
            predictions, labels = eval_preds
            start_logits, end_logits = predictions
            predicted_starts = np.argmax(start_logits, axis=1)
            predicted_ends = np.argmax(end_logits, axis=1)
            actual_starts = labels[0]
            actual_ends = labels[1]
            start_accuracy = (predicted_starts == actual_starts).mean()
            end_accuracy = (predicted_ends == actual_ends).mean()
            return {
                "start_accuracy": float(start_accuracy),
                "end_accuracy": float(end_accuracy),
                "average_accuracy": float((start_accuracy + end_accuracy) / 2),
            }
        except Exception as e:
            logging.error(f"Error in metrics computation: {str(e)}")
            return None

    def run_single_experiment(self, model_info: Dict, dataset_info: Dict) -> bool:
        experiment_name = f"{model_info['name']}_{dataset_info['name']}"
        logging.info(f"Starting experiment: {experiment_name}")

        try:
            # Load dataset
            dataset = load_dataset(dataset_info["name"].lower())
            small_train_dataset = dataset["train"].select(range(100))
            small_val_dataset = dataset["validation"].select(range(50))

            # Load model and tokenizer
            model_id = (
                model_info["huggingface_url"].split("/")[-2]
                + "/"
                + model_info["huggingface_url"].split("/")[-1]
            )
            tokenizer = AutoTokenizer.from_pretrained(model_id)
            model = AutoModelForQuestionAnswering.from_pretrained(model_id)

            # Preprocess datasets
            tokenized_train = small_train_dataset.map(
                lambda x: self.preprocess_squad(x, tokenizer),
                batched=True,
                remove_columns=small_train_dataset.column_names,
            )

            tokenized_val = small_val_dataset.map(
                lambda x: self.preprocess_squad(x, tokenizer),
                batched=True,
                remove_columns=small_val_dataset.column_names,
            )

            # Training arguments
            training_args = TrainingArguments(
                output_dir=f"{self.results_dir}/{experiment_name}",
                evaluation_strategy="epoch",
                learning_rate=2e-5,
                per_device_train_batch_size=4,  # Reduced batch size
                per_device_eval_batch_size=4,  # Reduced batch size
                num_train_epochs=1,
                weight_decay=0.01,
                push_to_hub=False,
                logging_dir=f"{self.results_dir}/{experiment_name}/logs",
                logging_steps=10,
                save_strategy="no",  # Don't save checkpoints to save space
            )

            # Initialize trainer
            trainer = Trainer(
                model=model,
                args=training_args,
                train_dataset=tokenized_train,
                eval_dataset=tokenized_val,
                tokenizer=tokenizer,
                data_collator=default_data_collator,
                compute_metrics=self.compute_metrics,
            )

            # Training
            train_results = trainer.train()
            eval_results = trainer.evaluate()

            # Save results
            results = {
                "model": model_info["name"],
                "dataset": dataset_info["name"],
                "training_loss": float(train_results.training_loss),
                "eval_results": eval_results,
                "timestamp": datetime.datetime.now().isoformat(),
            }

            with open(f"{self.results_dir}/{experiment_name}_results.json", "w") as f:
                json.dump(results, f, indent=2)

            self.successful_experiments.append(experiment_name)
            return True

        except Exception as e:
            error_msg = f"Error in experiment {experiment_name}: {str(e)}\n{traceback.format_exc()}"
            logging.error(error_msg)
            self.failed_experiments.append(
                {"experiment": experiment_name, "error": str(e)}
            )
            return False
        finally:
            self.cleanup_gpu()

    def run_all_experiments(self):
        total_experiments = len(self.models) * len(self.datasets)
        completed = 0
        start_time = datetime.datetime.now()

        for model in self.models:
            for dataset in self.datasets:
                completed += 1
                logging.info(f"Progress: {completed}/{total_experiments} experiments")

                success = self.run_single_experiment(model, dataset)

                if completed % 10 == 0:  # Send progress email every 10 experiments
                    self.send_progress_email(completed, total_experiments, start_time)

        self.send_final_report(start_time)

    def send_progress_email(
        self, completed: int, total: int, start_time: datetime.datetime
    ):
        current_time = datetime.datetime.now()
        elapsed_time = current_time - start_time
        avg_time_per_exp = elapsed_time / completed
        remaining_experiments = total - completed
        estimated_remaining_time = avg_time_per_exp * remaining_experiments

        content = f"""
        Progress Report

        Completed: {completed}/{total} experiments
        Success Rate: {len(self.successful_experiments)}/{completed}
        Failed Experiments: {len(self.failed_experiments)}
        
        Time Elapsed: {elapsed_time}
        Estimated Time Remaining: {estimated_remaining_time}
        """

        self.send_email("Experiment Progress Update", content)

    def send_final_report(self, start_time: datetime.datetime):
        end_time = datetime.datetime.now()
        total_time = end_time - start_time

        content = f"""
        Final Experiment Report

        Total Experiments: {len(self.models) * len(self.datasets)}
        Successful Experiments: {len(self.successful_experiments)}
        Failed Experiments: {len(self.failed_experiments)}
        
        Total Time: {total_time}
        Average Time per Experiment: {total_time / (len(self.models) * len(self.datasets))}

        Failed Experiments Details:
        {json.dumps(self.failed_experiments, indent=2)}
        """

        self.send_email("Experiment Final Report", content)


def main():
    parser = argparse.ArgumentParser(description="Run multiple fine-tuning experiments")
    parser.add_argument(
        "--email", type=str, required=True, help="Email for notifications"
    )
    args = parser.parse_args()

    manager = ExperimentManager(args.email)
    manager.run_all_experiments()


if __name__ == "__main__":
    main()
