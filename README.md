# TopSpin

A minimalistic library for training text-only dense LLMs on a single NVIDIA node (8 GPUs).


## Getting Started

### Installation

```bash
git clone https://github.com/satyaki-up/topspin.git
cd topspin
pip install -r requirements.txt
```

### Data Prep

```bash
# Download and tokenize the full dataset (1M samples)
python dataprocessing/download_data.py

# Or download a smaller subset for testing
python dataprocessing/download_data.py --max_samples 1000
```

Check configs/data.yaml for more details.

### Training

```bash
# Default
python trainer/train.py --data_path data/minipile_tokenized.pt

# Custom
python trainer/train.py --data_path data/minipile_tokenized.pt --config configs/model.yaml
```

Check configs/model.yaml for more details.