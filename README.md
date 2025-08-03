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
python dataprocessing/download_data.py --max_samples 1000000 --shard_size 10000
```

Check configs/data.yaml for more details.

### Training

```bash
python trainer/train.py --data_path data
```

Check configs/model.yaml for more details.