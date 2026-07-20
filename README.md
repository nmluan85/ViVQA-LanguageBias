# ViVQA-LanguageBias

# Counterfactual Object-Aware Integration for Vietnamese Visual Question Answering

This repository contains the official implementation for a robust Vietnamese Visual Question Answering (ViVQA) framework equipped with **Counterfactual Sample Synthesizing (CSS)** to handle language bias.

Since extracting features end-to-end requires a lot of computing power, we extract the local region and global features beforehand so they can be loaded directly during training.

---

## 📂 Repository Structure

```text
ViVQA-LanguageBias/
├── css/                     # Core Counterfactual Situation Supervision modules
│   ├── qcss.py              # Question-level CSS modifications
│   └── vcss.py              # Visual-level CSS modifications
├── css-candidates/          # Synthesized VCSS and QCSS of train, test, and val set
├── data/                    # Dataset root directory
│   ├── ViVQA-csv/           # Split dataset annotations (train.csv, val.csv, test.csv)
│   ├── images/              # Raw image files
│   └── vocab.json           # Answer vocabulary dictionary
├── features/                # Pre-extracted global (BLIP-2) and local (VinVL) visual     
├── output/                  # Trained checkpoints and model assessment binaries
├── baseline_model.py        # Standard VQA model architecture configuration
├── proposed_model.py        # Vietnamese VQA architecture without CSS
├── css_trainer.py           # ViVQA with Counterfactual Sample Synthesizing
├── dataset.py / dataset_css.py  # Data loaders and custom collation wrappers for CSS
├── extract_vision.py        # Feature extractor script (BLIP-2 / VinVL)
├── main.py / run_model.py   # Test a sample
├── train.py                 # Core model training script
├── eval.py                  # Evaluation suite script
├── processor.py             # Image Processor of BLIP
├── TextEmbedding.py         # Load BARTPho to extract word embedding
├── VisionEmbedding.py       # Load BLIP-2, VinVL to extract visual embedding.
└── requirements.txt         # Project dependencies checklist
```

---

## Quick Start

### 1. Environment Setup
Initialize your environment wrapper and install the required dependencies:
```bash
# Create and activate a python virtual environment
python -m venv .venv
source .venv/bin/activate  # On Windows use: .venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Feature Extraction (Preprocessing)
Because training models end-to-end with deep vision backbones requires substantial computing overhead, you must extract global attributes (**BLIP-2**) and regional local attributes (**VinVL**) beforehand.

Place your raw image instances in `data/images/` and execute:
```bash
python extract_vision.py
```
Extracted embeddings will automatically route to the `features/` directory to be consumed directly during training loops (blip-2.hdf5 for BLIP-2 and VinVL.hdf5 for VinVL).

Extracted features are available in 2 links:

**BLIP-2:** [\[link\]](https://drive.google.com/file/d/1Q8ZQRR6Sunz5SS-HjFvb1lFGPPuNxSar/view?usp=sharing)
**VinVL:** [\[link\]](https://drive.google.com/file/d/1Y8XGRe7JOvFFBE6JpNpKXe6bWkzMb5Gi/view?usp=sharing)

After downloading, please place them into the `./features/` directory.

---

## Training the Model

Run `train.py` to initiate the training loop. The training setup integrates both Question CSS (`--use-qcss`) and Visual CSS (`--use-vcss`) to prevent the model from memorizing language-bias correlations.

### Example Run
```bash
python train.py \
    --output-dir "./output" \
    --epochs 20 \
    --train-batch-size 8 \
    --gradient-accumulation-steps 4 \
    --learning-rate 3e-5
```

### Key Configurable Arguments

| Argument | Type | Default | Description |
| :--- | :---: | :---: | :--- |
| `--output-dir` | `str` | `./output` | Output directory for logs and check-pointed binaries. |
| `--lr-scheduler-type` | `str` | `cosine` | Learning rate scheduler style (`cosine` or `linear`). |
| `-tb`, `--train-batch-size` | `int` | `8` | Training mini-batch size. |
| `--gradient-accumulation-steps`| `int` | `4` | Number of backward passes to accumulate before an optimizer step. |
| `-e`, `--epochs` | `int` | `20` | Total training epochs. |
| `--use-vcss` / `--use-qcss` | `bool`| `True` | Flags to toggle Counterfactual Situation Supervision. |
| `--css-types` | `list`| `['counting', 'color']` | Target domains for generating counterfactual mutations. |
| `--alpha-css` | `float`| `0.1` | Regularization loss weight for the CSS objectives. |

For a complete breakdown of configuration settings, please reference `InstallationGuide.txt`.

---

## Evaluation

Model weight of paper available: [\[link\]](https://drive.google.com/file/d/1FBS5FJyP5zYWgjZrMoYd7_7JxMrwKSOa/view?usp=sharing). After downloading, please put it in folder ./output/

Evaluate the performance of trained models on your specified test partition by pointing `eval.py` to the saved binary weights file:

```bash
python eval.py --ckpt-path "./output/pytorch_model.bin"
```
