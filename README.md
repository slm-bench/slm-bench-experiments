# SLM-Bench: A Comprehensive Benchmark of Small Language Models on Environmental Impacts

[![License: Apache 2.0](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)

## Abstract

Small Language Models (SLMs) offer computational efficiency and accessibility, yet a systematic evaluation of their performance and environmental impact remains lacking. We introduce SLM-Bench, the first benchmark specifically designed to assess SLMs across multiple dimensions, including accuracy, computational efficiency, and sustainability metrics. SLM-Bench evaluates 12 SLMs on 9 NLP tasks using 23 datasets spanning 14 domains, providing a rigorous comparison of their effectiveness. Unlike prior benchmarks, SLM-Bench quantifies 11 metrics across correctness, computation, and consumption, enabling a holistic assessment of efficiency trade-offs. Our evaluation considers controlled hardware conditions, ensuring fair comparisons across models. We develop an open-source benchmarking pipeline with standardized evaluation protocols to facilitate reproducibility and further research. Our findings highlight the diverse trade-offs among SLMs, where some models excel in accuracy while others achieve superior energy efficiency. SLM-Bench sets a new standard for SLM evaluation, bridging the gap between resource efficiency and real-world applicability.

## Introduction

Recent advancements in Language Models (LMs) have revolutionized NLP applications across domains. However, the impressive performance of Large Language Models (LLMs) comes with significant computational costs and environmental challenges. Small Language Models (SLMs) have emerged as a promising solution to mitigate these negative impacts by significantly reducing the number of parameters while maintaining reasonable performance.

Despite their growing prominence, a notable gap exists in the systematic evaluation of SLMs. SLM-Bench addresses this gap by providing a comprehensive benchmarking framework for evaluating SLMs with a specific focus on their environmental impacts.

![alt text](https://github.com/slm-bench/slm-bench-experiments/blob/main/intro.png)

## Key Features

- **Focus on Small Language Models**: Evaluates SLMs (models with less than 7B parameters) that are computationally efficient and accessible to a wider range of users and systems
- **Measurement of Environmental Impacts**: Integrates metrics for energy consumption and CO2 emissions to assess sustainability, an aspect often overlooked in benchmarking
- **Evaluation Across Diverse Settings**: Rigorously evaluates 12 SLMs on 9 tasks using 23 datasets from 14 domains

## Datasets

SLM-Bench incorporates 23 diverse datasets spanning 14 domains and 9 task types:

| Dataset | #Samples | Domain | Task |
|---------|----------|--------|------|
| BoolQ | 15,432 | Open-domain | Question Answering |
| ARC-Easy | 5,876 | Open-domain | Question Answering |
| ARC-Challenge | 2,590 | Open-domain | Question Answering |
| OpenBookQA | 5,957 | Open-domain | Question Answering |
| PIQA | 16,113 | Physics | Reasoning |
| Hellaswag | 10,421 | Common Sense | Reasoning |
| WinoGrande | 44,321 | Common Sense | Reasoning |
| CommonsenseQA | 12,102 | Common Sense | Reasoning |
| GSM8k | 8,034 | Mathematics | Problem Solving |
| AQuA | 99,765 | Mathematics | Problem Solving |
| RACE-Middle | 24,798 | Education | Reading Comprehension |
| RACE-High | 26,982 | Education | Reading Comprehension |
| CoQA | 127,542 | Open-domain | Question Answering |
| e2e_nlg | 50,321 | Food & Beverage | Text Generation |
| viggo | 9,842 | Video Games | Text Generation |
| glue_qnli | 104,543 | Linguistics | Question Answering |
| bc5cdr | 20,764 | Chemistry | Recognition |
| conllpp | 23,499 | Linguistics | Recognition |
| customer_support | 14,872 | Customer Behaviors | Classification |
| legal | 49,756 | Legal | Classification |
| reuters | 9,623 | News | Topic Extraction |
| covid | 19,874 | Healthcare | Sentiment Analysis |
| drop | 96,567 | Open-domain | Reasoning |

## Models

SLM-Bench evaluates 12 SLMs selected based on model size (<7B parameters), popularity, and open-source availability:

| Model | Provider | #Params (B) | Size (GB) | Year |
|-------|----------|-------------|-----------|------|
| GPT-Neo-1.3B | EleutherAI | 1.37 | 2.46 | 03/2021 |
| Open-LLaMA-3B | OpenLM | 3 | 6.8 | 06/2024 |
| LLaMA-2-7B | Meta | 6.47 | 13 | 07/2023 |
| Mistral-7B | Mistral AI | 7 | 13 | 09/2023 |
| Zephyr-7B | WebPilot.AI | 7 | 13.74 | 11/2023 |
| Phi-1.5B | Microsoft | 2.7 | 2.45 | 12/2023 |
| TinyLlama-1.1B | SUTD | 1.1 | 2 | 08/2023 |
| StableLM-3B | Stability AI | 3 | 6.5 | 12/2023 |
| ShearedLlama-2.7B | Princeton NLP | 2.7 | 5 | 11/2023 |
| Dolly-v2-3B | Databricks | 3 | 5.8 | 12/2022 |
| Gemma-2B | Google | 2 | 4.67 | 11/2023 |
| Pythia-2.8B | EleutherAI | 2.8 | 5.5 | 02/2023 |

## Evaluation Metrics

SLM-Bench employs 11 metrics categorized into three groups:

1. **Correctness Evaluation**
   - Accuracy
   - F1 Score
   - BLEU
   - ROUGE
   - METEOR
   - Perplexity

2. **Computation Evaluation**
   - Runtime
   - FLOP (Floating Point Operations)

3. **Consumption Evaluation**
   - Cost (USD)
   - CO2 Emissions (kg)
   - Energy Usage (kWh)

![alt text](https://github.com/slm-bench/slm-bench-experiments/blob/main/result.png)

## Benchmarking Pipeline

Our benchmarking pipeline consists of 7 key modules:

1. **Universal Data Loader**: Converts datasets into a unified structure
2. **Preprocessing**: Refines data through trimming and transformation
3. **Calling**: Manages execution of SLMs and tasks
4. **Postprocessing**: Refines model outputs
5. **Evaluation**: Applies appropriate metrics
6. **Report**: Compiles results and visualizations
7. **Logging**: Records events throughout for traceability

## Key Findings

Our evaluation reveals several important insights:

- **Performance Trade-offs**: Different models excel in different aspects - some offer superior accuracy while others prioritize energy efficiency
- **Top Performers**:
  - **Correctness**: Zephyr-7B and Mistral-7B achieved the highest accuracy
  - **Computation**: GPT-Neo-1.3B and TinyLlama-1.1B demonstrated the best computational efficiency
  - **Consumption**: Phi-1.5B and StableLM-3B minimized resource consumption most effectively
- **Well-Balanced Models**: GPT-Neo-1.3B and ShearedLlama-2.7B maintained strong overall performance across all three categories

## Recommendations

Based on our findings, we recommend:

- For accuracy-focused applications: Zephyr-7B or Mistral-7B
- For computationally-constrained environments: Phi-1.5B or ShearedLlama-2.7B
- For minimizing environmental impact: Phi-1.5B or GPT-Neo-1.3B
- For balanced performance: GPT-Neo-1.3B or ShearedLlama-2.7B

## Installation & Usage

```bash
# Clone the repository
git clone https://github.com/slm-bench/slm-bench.git
cd slm-bench

# Install dependencies
pip install -r requirements.txt

# Run the benchmark with default settings
python run_benchmark.py --all

# Run a specific evaluation
python run_benchmark.py --model gpt-neo-1.3b --dataset boolq --task qa
```

## Citation

```bibtex
@inproceedings{author2024slmbench,
  title={SLM-Bench: A Comprehensive Benchmark of Small Language Models on Environmental Impacts},
  author={Author, Anonymous},
  booktitle={Proceedings of the Annual Meeting of the Association for Computational Linguistics},
  year={2024}
}
```

## License

This project is licensed under the Apache License 2.0 - see the [LICENSE](LICENSE) file for details.

## Links

- [SLM-Bench Source Code](https://anonymous.4open.science/r/slm-bench-experiments-87F6)
- [SLM-Bench Leaderboard](https://slm-bench.github.io/leaderboard)