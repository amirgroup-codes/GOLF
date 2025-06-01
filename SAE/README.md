# Sparse Autoencoder (SAE) Probing for OLF Variant Interpretation

This folder contains scripts and outputs related to probing and visualizing sparse autoencoder (SAE) representations of protein sequences for pathogenicity interpretation.

## Overview

We use a sparse autoencoder (SAE) trained on frozen ESM2 embeddings to learn interpretable representations of OLF variants. Linear probes are trained on SAE embeddings to predict EVE scores, and the most predictive latent dimensions are visualized structurally using PyMOL.

## Setup

Set up a conda environment with the required packages:

```bash
conda env create -f sae.yml
conda activate sae
```

Clone the InterProt repository (if not already present):

```bash
git clone https://github.com/liambai/InterProt.git
```

Ensure required models are downloaded (from HuggingFace) and placed in the `models/` folder:
  - [ESM2 model](https://huggingface.co/facebook/esm2_t33_650M_UR50D/tree/main) (`esm2_t33_650M_UR50D`) 
  - [SAE model](https://huggingface.co/liambai/InterProt-ESM2-SAEs/blob/main/esm2_plm1280_l24_sae4096.safetensors) (`esm2_plm1280_l24_sae4096.safetensors`)

Ensure you have files `models/models--facebook--esm2_t33_650M_UR50D/snapshots/08e4846e537177426273712802403f7ba8261b6c/model.safetensors` and `models/models--liambai--InterProt-ESM2-SAEs/snapshots/3ded2e51641df84d6b89f25430874e3a8eb42c24/esm2_plm1280_l24_sae4096.safetensors`.

## Step 1: Linear Probe on SAE Embeddings

To extract SAE embeddings and train a linear probe to predict EVE scores:

```bash
python probe_sae.py
```

This will:
- Compute SAE activations from mean-pooled ESM2 embeddings (layer 24)
- Train a ridge regression model to predict EVE scores
- Save sorted SAE weights into `results/weights/sae_raw_layer24.csv`

## Step 2: Visualize Predictive Latents

To visualize the most predictive SAE latent dimensions:

```bash
python visualize_sae.py
```

This will:
- Identify peaks in SAE activations for the input sequence
- Highlight the top positive (pathogenic) and negative (benign) units
- Save a PyMOL script (`highlight_units_layer24.pml`)
- Save residue-level summaries per latent (`highlighted_latents_layer24.txt`)

## Description of Files

| File                              | Description                                                                 |
|-----------------------------------|-----------------------------------------------------------------------------|
| `interprot/`                      | SAE-related modules cloned from the InterProt GitHub repository            |
| `models/`                         | Folder containing ESM2 and SAE models|
| `results/`                        | Output directory with sorted SAE probe weights                             |
| `highlight_units_layer24.pml`     | PyMOL script highlighting top latent-residue associations                   |
| `highlighted_latents_layer24.txt`| List of residues associated with top latent units                          |
| `mutated_sequences_with_scores.csv` | Input CSV file with sequences and EVE scores                               |
| `probe_sae.py`                    | Script for embedding extraction and ridge regression probing               |
| `visualize_sae.py`                | Script to identify top latent units and generate visualizations            |

## Notes

- Positive weights correspond to pathogenic-associated latents.
- Negative weights correspond to benign-associated latents.
