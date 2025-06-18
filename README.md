# LLM Acceleration

## Overview
This project explores how to accelerate **LLaMA-3.2-3B-Instruct** inference on a single **NVIDIA T4 GPU**, while keeping **perplexity under or equal to 11.5**.

The optimization pipeline includes:

1. LoRA fine-tuning on the 3B teacher model  
2. Knowledge distillation from 3B to 1B using GKD  
3. LoRA fine-tuning on the distilled 1B student model  
4. Quantization using [HQQ](https://github.com/mobiusml/hqq) with hybrid 4/8-bit precision  
5. Inference backend optimization with `torch.compile` and `gemlite`

## Setup and Reproduce
First, clone the repository:

```bash
git clone https://github.com/bonginn/llm-acceleration.git
cd llm-acceleration
```

We **strongly recommend using a Conda environment** to avoid dependency conflicts and ensure reproducibility:

```bash
conda create -n llama-acc python=3.10 -y
conda activate llama-acc
```
Then install the required packages:
```bash
pip install torch==2.7.0
pip install -r requirements.txt
```
> ⚠️ **Note:** `torch` must be installed **before** `gemlite` since `gemlite` imports `torch` during installation.

After installing all dependencies, make sure you are logged in to Hugging Face to access the LLaMA models:

```bash
huggingface-cli login
```
> You can generate one at [Hugging Face](https://huggingface.co/settings/tokens).

Running inference to reproduce the results:
```bash
python3 inference.py
```

If you want to run distillation and/or LoRA fine-tuning, you can use the following commands:

```bash
python3 distillation.py
python3 lora.py
```
> It may take ~10 hours to run distillation and ~1.5 hours to run LoRA fine-tuning on a single A100 40GB.

## Results

| Model Configuration | Perplexity | Throughput (tokens/s) |
|-------------------|------------|-----------------|
| 3B Base | 11.05 | 26.2 |
| 3B + LoRA | 9.67 | 26.2 |
| Distilled 1B | 11.76 | 67.5 |
| Distilled 1B + LoRA | 11.42 | 67.5 |
| Distilled 1B + LoRA + HQQ | 11.49 | 125.0 |

> Note: All experiments were conducted on a single NVIDIA T4 GPU.
> The throughput measurements are end-to-end (e2e), including both prefill and decoding phases.
    