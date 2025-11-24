# Melody: Decoding the Sequence Determinants of Locus-Specific DNA Methylation Across Human Tissues
This repository contains the official implementation of Melody, a deep learning framework designed to decipher the DNA sequence determinants underlying human DNA methylation landscapes. Melody accurately predicts cell-type-specific methylation profiles and generalizes to unseen cell types via scRNA-seq integration.



## 🔍 Overview



Melody leverages a specialized U-Net architecture with a large receptive field (10kb) to capture long-range genomic dependencies. It supports:

1. **Locus-Specific Prediction:** Accurate methylation level prediction at single-CpG resolution.
2. **Multi-Task Learning:** Simultaneously predicts methylation levels, CpG counts, and regional averages.
3. **Cross-Modal Generalization:** Predicts methylation for **unseen cell types** using scRNA-seq foundation models (Melody-G).



## 🧩 Framework Variants



The code supports three variants of the model:

- **Melody-ST (Single-Track):** Specialized for a single cell type.
- **Melody-MT (Multi-Track):** Jointly models methylation across multiple tissues (e.g., 39 cell types).
- **Melody-G (Generalize):** Integrating scRNA-seq embeddings to predict methylation in unseen cell types.



## 🛠️ Installation

We recommend using Anaconda or Miniconda for environment management.

### 1. Create Environment

```bash
conda create -n melody python=3.10
conda activate melody
```

### 2. Install PyTorch

Choose the command that matches your CUDA version (check with `nvidia-smi`).

*For CUDA 11.8:*

```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

*For newer CUDA versions:*

```bash
pip install torch torchvision torchaudio
```

### 3. Install Dependencies

```bash
# Core libraries
pip install transformers einops ninja seaborn loguru echo_logger
pip install scikit-learn tensorboard matplotlib jupyter tqdm pandas accelerate fire
conda install zlib

# Genomics & Visualization tools
pip install h5py h5sparse pyBigWig tensorboardX medpy pytabix pyfaidx wandb plotly liftover
pip install selene-sdk

# Clean up
pip cache purge
```

*Note: You may need to install `cupy-cuda11x` or `cupy-cuda12x` depending on your driver version if required by specific sub-dependencies.*

## 📂 Data Preparation

Please download the necessary dataset (Reference Genome `hg38`, processed BigWigs, and Checkpoints) from our Google Drive.

**[📥 Download Data (Google Drive)](https://drive.google.com/drive/folders/1O1OZ_w-3X97MM47XSmgc_165n2dJ1KND?usp=sharing)**

Organize the downloaded files into a `data/` directory in the project root:

```
Melody/
├── data/
│   ├── fasta/
│   │   └── Homo_sapiens.GRCh38.dna.primary_assembly.fa
│   ├── bigwigs/
│   │   ├── GSM5652317_Blood-B-Z000000UB.hg38.bigwig
│   │   ├── ... (other cell types)
│   └── checkpoints/
├── melodyG1/
├── melodyG2/
├── run.py
└── ...
```


## 🚀 Usage

The main training script is `run.py`. It automatically switches between **Melody-ST** and **Melody-MT** modes based on the number of BigWig files provided.

### Training Melody-ST & Melody-MT

1. Melody-ST (Single Track)

Train on a specific cell type (e.g., Blood-B).

```bash
python run.py \
  --lab_name "Melody_ST_Blood" \
  --bigwigs_files "GSM5652317_Blood-B-Z000000UB.hg38.bigwig" \
  --gpu 0 \
  --window_size 10000 \
  --batch_size 32 \
  --lr 0.001
```

2. Melody-MT (Multi Track)

Train jointly on multiple cell types. Simply provide multiple files or use a directory scan logic if implemented in your custom wrapper.

```bash
python run.py \
  --lab_name "Melody_MT_All" \
  --bigwigs_files "file1.bigwig" "file2.bigwig" ... \
  --use_cg_loss \
  --use_avg_loss \
  --gpu 0
```



### Training Melody-G

Melody-G involves a two-stage training process located in subdirectories:

- **Stage 1 (G1):** Pre-training on whole chromosomes.

  ```bash
  cd melodyG1
  python run.py --lab_name "Melody_G1_Pretrain" ...
  ```

- **Stage 2 (G2):** Fine-tuning on cell-type-specific regions.

  ```bash
  cd melodyG2
  python run.py --one_stage_ckpt "../path/to/stage1.ckpt" ...
  ```



### Key Arguments

| **Argument**                   | **Default** | **Description**                                     |
|--------------------------------|-------------|-----------------------------------------------------|
| `--window_size`                | 10000       | Input DNA sequence length (Receptive field).        |
| `--bigwigs_files`              | ...         | List of target BigWig files for training.           |
| `--any_cpg_focal_weight`       | 8.0         | Weight for any CpG sites.                           |
| `--low_methy_cpg_focal_weight` | 32.0        | Weight for low-methylation sites                    |
| `--use_cg_loss`                | False       | Enable auxiliary CpG count prediction loss.         |
| `--use_avg_loss`               | False       | Enable auxiliary regional average methylation loss. |



## 📊 Monitoring

This project uses [WandB](https://www.google.com/search?q=https://wandb.ai/) for logging. Ensure you are logged in:

```bash
wandb login
```

Training logs (Loss, LR, etc.) will be synced to your WandB project defined by `--project` (Default: "Melody").



## 🖊️ Citation

If you find this work useful for your research, please cite our paper:

```
@article{Melody2025,
  title={Melody: Decoding the Sequence Determinants of Locus-Specific DNA Methylation Across Human Tissues},
  author={Jin, Junru and Wang, Ding and Qiao, Jianbo and Gao, Wenjia and Liu, Yuhang and Chen, Siqi and Zou, Quan and Wu, Shu and Su, Ran and Wei, Leyi},
  journal={bioRxiv},
  year={2025}
}
```



## 📧 Contact

For any questions, please open an issue or contact the authors.
