# LLM Acceleration

This project explores how to accelerate **LLaMA-3.2-3B-Instruct** inference on a single **NVIDIA T4 GPU**, while keeping **perplexity under or equal to 11.5**.

The optimization pipeline includes:

1. LoRA fine-tuning on the 3B teacher model  
2. Knowledge distillation from 3B to 1B using GKD  
3. LoRA fine-tuning on the distilled 1B student model  
4. Quantization using [HQQ](https://github.com/mobiusml/hqq) with hybrid 4/8-bit precision  
5. Inference backend optimization with `torch.compile` and `gemlite`

---

## Usage
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

