# KITE: Korean Instruction-following Task Evaluation

[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![Dataset](https://img.shields.io/badge/🤗%20Dataset-KITE-yellow)](https://huggingface.co/datasets/junkim100/KITE)
[![arXiv](https://img.shields.io/badge/arXiv-2510.15558-b31b1b.svg)](https://arxiv.org/abs/2510.15558)

## Overview

**KITE (Korean Instruction-following Task Evaluation)** is the first comprehensive benchmark specifically designed to evaluate the Korean instruction-following capabilities of Large Language Models (LLMs). Unlike existing Korean benchmarks that focus mainly on factual knowledge or multiple-choice testing, KITE directly targets diverse, open-ended instruction-following tasks.

The instruction-following capabilities of LLMs are pivotal for numerous applications, from conversational agents to complex reasoning systems. However, current evaluations predominantly focus on English models, neglecting the linguistic and cultural nuances of other languages. KITE addresses this gap by providing a benchmark that captures the unique characteristics of the Korean language, including its complex syntax, rich morphological features, honorific system, and dual numbering systems.

## Key Features

- **Comprehensive Evaluation**: Two distinct benchmarks covering both general and Korean-specific instruction-following tasks
- **KITE General**: 427 instructions translated and filtered from Google's IFEval dataset
- **KITE Korean**: 100 instructions created from scratch to address Korean-specific linguistic features
- **Verifiable Instructions**: Rule-based evaluation ensuring objective and measurable outcomes
- **Cultural Awareness**: Instructions embedded with Korean cultural context

## KITE Benchmark Structure

### KITE General (427 Instructions)

KITE General consists of universally applicable tasks derived from the IFEval dataset. The development process involves:

1. **Automated Translation**: Using GPT-4o to translate the original IFEval dataset
2. **Manual Verification**: Meticulous review to identify and correct translation errors
3. **Contextual Filtering**: Removal of English-centric instructions (e.g., capitalization, English-only responses)
4. **Expert Review**: Five native Korean speakers with NLP knowledge reviewed all instructions

Out of the original 541 IFEval instructions, 114 were filtered out as culturally or linguistically irrelevant, resulting in 427 high-quality Korean instructions.

### KITE Korean (100 Instructions)

KITE Korean comprises 100 instructions (25 per category) created from scratch to evaluate Korean-specific linguistic phenomena:

| Category | Description | Example |
|----------|-------------|---------|
| **Acrostic Poem (삼행시)** | Generate structured poetry where each line starts with a specific letter from a given word | "Write an acrostic poem using the word '밤하늘' (night sky)" |
| **Post-position Drop (조사 생략)** | Form sentences without Korean grammatical markers (postpositions) while preserving meaning | "Explain the origin of the Korean script without using subject or object postpositions" |
| **Honorifics (존댓말/반말)** | Switch between honorific and informal speech styles | "Convert the following sentence to informal speech: '어제 정말 즐거웠어요. 다음에 또 만나요.'" |
| **Native/Sino Korean Numbers (순한국어/한자어 숫자)** | Convert between native Korean and Sino-Korean number systems | "Change the numbers in the following sentence to native Korean: '이 회의는 90분 동안 지속됩니다.'" |

## Why KITE?

Korean presents unique challenges for LLMs:

- **Agglutinative Structure**: Combination of roots and affixes to form words and sentences
- **Flexible Word Order**: Post-positions (Josa) provide syntactic information, allowing flexible word order
- **No Case Distinction**: Unlike English, Korean does not distinguish between uppercase and lowercase
- **Honorific System**: Complex levels of politeness embedded in grammar
- **Dual Number Systems**: Native Korean and Sino-Korean numbers used in different contexts
- **Cultural Context**: Communication styles deeply embedded in Korean culture

Existing English-centric benchmarks fail to capture these nuances, making KITE essential for accurate evaluation of Korean LLMs.

## Installation

### Prerequisites

- Python 3.9 or higher
- CUDA-compatible GPU (for HuggingFace models)
- Conda (recommended)

### Setup

1. Clone the repository:
```bash
git clone https://github.com/junkim100/KITE.git
cd KITE
```

2. Create and activate the conda environment:
```bash
conda env create -f kite.yml
conda activate kite
```

3. Install the package:
```bash
cd korean_instruction_following_eval
pip install -e .
```

## Quick Start

### Running Evaluation

The easiest way to run evaluation is using the provided `run.sh` script:

```bash
bash run.sh
```

### Configuration

Edit `run.sh` to configure your evaluation:

```bash
# Dataset selection: 'general' or 'korean'
DATASET_TYPE="korean"

# For KITE Korean, select categories
KOREAN_CATEGORIES=('acrostic' 'honorifics' 'numbers' 'postposition')

# Model configuration
MODEL_TYPE="hf"  # Options: 'openai', 'hf', 'solar', 'clova'
MODELS=("meta-llama/Meta-Llama-3-8B-Instruct")

# Shot configuration
SHOT_NUM=0  # Options: 0, 1, 3, 5
```

### Manual Evaluation

You can also run evaluation manually using Python:

```bash
python korean_instruction_following_eval/main.py \
  --instruction_file korean_instruction_following_eval/data/culturally_aware/instruction/acrostic.jsonl \
  --response_output_dir korean_instruction_following_eval/data/culturally_aware/response/acrostic/0_shot \
  --eval_output_dir korean_instruction_following_eval/data/eval_results/acrostic/0_shot \
  --shot_num 0 \
  --verbosity -1 \
  --model_type hf \
  --model meta-llama/Meta-Llama-3-8B-Instruct
```

## Supported Models

### OpenAI Models
- `gpt-3.5-turbo`
- `gpt-4o`

### HuggingFace Models
- `meta-llama/Meta-Llama-3-8B-Instruct`
- `google/gemma-7b-it`
- `yanolja/EEVE-Korean-Instruct-10.8B-v1.0`
- Any other HuggingFace model with instruction-following capabilities

### Korean-Specific Models
- SOLAR 1 Mini Chat (`solar-1-mini-chat`)
- HyperCLOVA X (`HPX-3.0`)
- EEVE Korean models

## Evaluation Methodology

KITE employs **verifiable instructions** to ensure clear and measurable outcomes. Each instruction is decomposed into sub-instructions, and the model's response is evaluated for each sub-instruction using rule-based checking.

### Accuracy Calculation

```
Accuracy = (Σ Σ f(s_ij)) / (Σ n_i) × 100%
```

Where:
- `N` = total number of instructions
- `n_i` = number of sub-instructions for instruction i
- `f(s_ij)` = 1 if sub-instruction s_ij is followed correctly, 0 otherwise

## Benchmark Results

Performance of various models on KITE (0-shot setting):

| Model | KITE General | KITE Korean |
|-------|--------------|-------------|
| GPT-4o | **89.35%** | **61.42%** |
| GPT-3.5-turbo | 75.92% | 46.19% |
| Llama 3 8B Instruct | 70.83% | 51.77% |
| Gemma 7b Instruct | 76.85% | 48.73% |
| EEVE v1.0 10.8b Instruct | 75.92% | 48.73% |
| HyperCLOVA X 003 | 60.64% | 45.68% |
| SOLAR 1 Mini Chat | 46.29% | 32.99% |

**Key Findings**:
- GPT-4o demonstrates the highest performance across both benchmarks
- Korean-specific models still lag behind GPT-4 in Korean language proficiency
- Performance varies significantly across different shot settings
- Instruction-following requires specialized tuning beyond general language understanding

## Dataset Structure

```
korean_instruction_following_eval/
├── data/
│   ├── culturally_aware/          # KITE Korean
│   │   └── instruction/
│   │       ├── acrostic.jsonl     # 25 acrostic poem instructions
│   │       ├── honorifics.jsonl   # 25 honorifics instructions
│   │       ├── numbers.jsonl      # 25 number system instructions
│   │       ├── postposition.jsonl # 25 postposition instructions
│   │       └── culturally_aware.jsonl  # All 100 combined
│   └── translated_and_filtered/   # KITE General
│       └── instruction/
│           └── relevant.jsonl     # 427 translated instructions
├── eval/                          # Evaluation logic
│   ├── evaluation_main.py
│   ├── instructions.py
│   ├── instructions_registry.py
│   └── instructions_util.py
└── main.py                        # Entry point
```

## Citation

If you use KITE in your research, please cite:

```bibtex
@misc{kim2025kitebenchmarkevaluatingkorean,
      title={KITE: A Benchmark for Evaluating Korean Instruction-Following Abilities in Large Language Models}, 
      author={Dongjun Kim and Chanhee Park and Chanjun Park and Heuiseok Lim},
      year={2025},
      eprint={2510.15558},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2510.15558}, 
}
```

## Contributing

We welcome contributions to KITE! Please feel free to:
- Report bugs or issues
- Suggest new Korean-specific instruction categories
- Improve evaluation metrics
- Add support for new models

## License

This project is licensed under the Apache License 2.0 - see the LICENSE file for details.

## Acknowledgments

- Based on Google's IFEval benchmark
- Thanks to all native Korean speakers who participated in the filtering and evaluation process
- Supported by the Korean NLP research community

## Contact
- **Author**: Dongjun Kim, Chanhee Park, Chanjun Park, Heuiseok Lim
- For questions or feedback, please open an issue on GitHub or contact Dongjun Kim: junkim100@gmail.com
