# GOLF: A Generative AI Framework for Pathogenicity Prediction for Myocilin OLF Variants

## Overview
This repository contains the codebase and results for GOLF:
1.  Multiple Sequence Alignment (MSA) generation from protein sequence data.
2.  Phylogenetic analysis to understand evolutionary relationships.
3.  Training and fine-tuning protein language models:
    *   Evolutionary model of Variant Effect (EVE) ensemble.
    *   ESM-1b model.
4.  Interpreting variant effects using Sparse Autoencoders (SAEs) on ESM2 embeddings.

## Workflow

The overall process is detailed below:

### 1. Data Collection and MSA Generation
*   **Sequence Collection**: Homologous sequences are collected using `jackhmmer` against the UniRef50 database.
*   **MSA Creation**: After removing duplicate sequences, `MMSeqs2` (specifically the `easy-cluster` command) is employed to generate a Multiple Sequence Alignment (MSA). This MSA is constructed with a sequence identity threshold of 95% and coverage of 80%.
*   This MSA serves as the primary input for training the EVE models and fine-tuning the ESM-1b model.

### 2. Phylogenetic Analysis
*   A phylogenetic tree is constructed to analyze the evolutionary context of the sequences, using an MSA clustered at 80% sequence identity.
*   The `phylo/` directory houses all relevant scripts and data for this stage.
*   **Tree Construction**: [IQ-TREE](https://iqtree.github.io/) is used for building the phylogenetic tree. The specific command can be found in `phylo/README.md`.
    *   Input: `phylo/MSA_80cluster.fas`
    *   Output: `phylo/MSA_80cluster.treefile` (raw tree), `phylo/MSA_80cluster_cleaned.treefile` (processed tree).
*   **Processing and Visualization**: The script `phylo/clean_tree.py` standardizes leaf names in the tree and generates annotation files (e.g., `phylo/dataset_colorstrip.txt`) for enhanced visualization with [iTOL (Interactive Tree Of Life)](https://itol.embl.de/).

### 3. EVE Model Training and Ensemble Analysis
*   The Evolutionary model of Variant Effect (EVE) is utilized to predict variant effects.
*   **Model Training**:
    *   The core Variational Autoencoder (VAE) for EVE models is trained using a script analogous to `train_VAE.py`, taking the MSA as input.
    *   Evolutionary indices, a key output from EVE models indicating variant impact, are computed from the VAE's latent space using a script like `compute_evol_indices.py`.
*   **Ensemble Creation**: An ensemble of EVE models is created by training multiple models, typically with different random seeds. These models, along with their configurations (`model_params.json`), checkpoints, logs, and computed `evol_indices`, are stored in subdirectories within `EVE Ensemble/` (e.g., `OLF-40_seed100_seed100_theta0.25_ld40_lr0.0001/`).
*   **Ensemble Performance Assessment**: The `Ensemble Analysis/` directory contains scripts for evaluating the EVE ensemble:
    *   `ensemble_evol_indices_analysis.py`: This script aggregates `evol_indices` from individual models in the ensemble. It then employs a Gaussian Mixture Model (GMM) analysis using `train_GMM_and_compute_EVE_scores.py` to convert evolutionary indices into pathogenicity scores and calculates prediction accuracy against a ground truth set of mutations.
    *   `visualize_ensemble_results.py`: Generates various plots to illustrate the ensemble's accuracy, its improvement over individual models, and other performance metrics.
    *   Analysis results, including plots and summary CSV files, are saved in this directory.
    *   Relevant data sources are found in `/EVE Data` to run the aformentioned scripts.

### 4. ESM-1b Fine-tuning
*   The ESM-1b protein language model is fine-tuned on the OLF MSA to adapt it for variant effect prediction.
*   **Fine-tuning Process**: The script `fine_tune_esm1b.py` manages this process. It loads the MSA, freezes the initial layers of the pre-trained ESM-1b model, and fine-tunes the subsequent layers.
*   **Outputs**: The fine-tuned model checkpoints (e.g., `esm1b_finetuned.pt`, `best_model.pt`), training logs, and related plots are stored in the `ESM/ESM1b/` directory.

### 5. Sparse Autoencoder (SAE) Analysis for Interpretation
*   To understand the features learned by large protein models and how they relate to variant effects, a Sparse Autoencoder (SAE) is applied to ESM2 embeddings.
*   The `SAE/` directory contains all scripts, configuration files, and detailed instructions for this analysis (see `SAE/README.md`).
*   **Probing SAE Latents**:
    *   `probe_sae.py`: This script computes SAE activations from mean-pooled ESM2 embeddings (specifically from layer 24) for a set of variants. It then trains a linear regression model (Ridge regression) to predict EVE scores based on these SAE activations. The weights of this linear model indicate which SAE latent dimensions are most predictive of pathogenicity.
*   **Visualization**:
    *   `visualize_sae.py`: Identifies the top SAE latent dimensions (both pathogenic and benign-associated) based on the probe weights. It generates:
        *   A PyMOL script (`highlight_units_layer24.pml`) to visualize these latents and their associated residues on the protein structure.
        *   A text file (`highlighted_latents_layer24.txt`) summarizing residue-level associations for the top latents.
*   Input data for this step typically includes a list of mutated sequences and their corresponding EVE scores (e.g., `SAE/mutated_sequences_with_scores.csv`).

## Repository Structure

Key directories and files in this repository:

*   `README.md`: This file.
*   `fine_tune_esm1b.py`: Script for fine-tuning the ESM-1b model.
*   `train_VAE.py`: Script for training the VAE component of EVE models.
*   `compute_evol_indices.py`: Script to compute evolutionary indices from trained EVE models.
*   `phylo/`: Contains scripts, data, and README for phylogenetic analysis.
    *   `clean_tree.py`: Processes phylogenetic trees and generates iTOL annotations.
*   `EVE Ensemble/`: Stores trained EVE models from multiple runs/seeds.
    *   Each subdirectory contains model parameters, checkpoints, logs, and evolutionary indices.
*   `Ensemble Analysis/`: Scripts and results for EVE ensemble performance analysis.
    *   `ensemble_evol_indices_analysis.py`: Core script for ensemble evaluation.
    *   `visualize_ensemble_results.py`: Generates plots for ensemble performance.
*   `ESM/`: Contains fine-tuned ESM model artifacts.
    *   `ESM1b/`: Fine-tuned ESM-1b model checkpoints, logs, and plots.
*   `SAE/`: Scripts, data, and detailed README for Sparse Autoencoder analysis.
    *   `sae.yml`: Conda environment definition for SAE tasks.
    *   `probe_sae.py`: Script for training a linear probe on SAE embeddings.
    *   `visualize_sae.py`: Script for visualizing predictive SAE latents.
*   `utils/`: Helper functions for EVE.
*   `data/`: Data used to construct MSA.
*   `examples/`: Example script runs for EVE.

## Setup and Installation

General setup guidelines. For specific modules like SAE, refer to their dedicated README files (e.g., `SAE/README.md`).

1.  **Core Tools**:
    *   Ensure `jackhmmr` (from the HMMER suite) and `MMSeqs2` are installed and available in your system's PATH.
2.  **Phylogenetic Analysis**:
    *   **IQ-TREE**: Required for phylogenetic tree construction. Download from the [IQ-TREE website](https://iqtree.github.io/).
    *   **Python**: A Python environment with `pandas` is needed for `phylo/clean_tree.py`.
3.  **EVE Model Training & Analysis**:
    *   The EVE framework relies on Python with standard scientific libraries (NumPy, Pandas) and PyTorch. Ensure these are installed.
    *   Refer to EVE documentation for specific version requirements if available.
4.  **ESM-1b Fine-tuning**:
    *   **Python Environment**: Requires PyTorch and the `esm` library by Facebook Research.
        ```bash
        pip install torch fair-esm matplotlib pandas tqdm
        ```
    *   **Hardware**: A GPU is highly recommended for efficient fine-tuning.
5.  **SAE Analysis**:
    *   A dedicated Conda environment is specified in `SAE/sae.yml`. Create and activate it:
        ```bash
        conda env create -f SAE/sae.yml
        conda activate sae
        ```
    *   Follow instructions in `SAE/README.md` to clone the `InterProt` repository and download necessary ESM2 and SAE models, placing them in `SAE/models/`.

## Usage

This section provides guidance on executing the different stages of the analysis pipeline.

### 1. Data Preparation
*   Use `jackhmmr` to search against UniRef50 and gather sequences.
*   Process the output to remove duplicates.
*   Use `MMSeqs2 easy-cluster` (with 95% identity, 80% coverage) to generate the MSA. This MSA (e.g., `your_msa.a3m` or `your_msa.fasta`) will be used in subsequent steps.
*   Prepare a version of the MSA for phylogenetic analysis (e.g., clustered at 80% identity, `phylo/MSA_80cluster.fas`).

### 2. Phylogenetic Analysis
1.  Navigate to the `phylo/` directory.
2.  Ensure your MSA for phylogeny (e.g., `MSA_80cluster.fas`) is present.
3.  Run IQ-TREE using the command specified in `phylo/README.md`.
4.  Execute `python clean_tree.py` to process the output tree and generate annotation files.
5.  Upload the cleaned treefile (e.g., `MSA_80cluster_cleaned.treefile`) and annotation files (e.g., `dataset_colorstrip.txt`) to iTOL for visualization.

### 3. EVE Training and Ensemble Analysis

*   **Training an EVE Model (Conceptual)**:
    ```bash
    # Train the VAE component
    python train_VAE.py --msa_file path/to/your_msa.a3m \\
                        --output_dir path/to/your_eve_model_dir \\
                        ...

    # Compute evolutionary indices
    python compute_evol_indices.py --model_checkpoint path/to/your_eve_model_dir/checkpoints/best_model.pt \\
                                   --msa_file path/to/your_msa.a3m \\
                                   --output_file path/to/your_eve_model_dir/evol_indices/evol_indices.csv \\
                                   ...
    ```
    Repeat with different seeds/configurations for ensemble members, storing outputs in `EVE Ensemble/`.

*   **Ensemble Analysis**:
    1.  Navigate to `Ensemble Analysis/`.
    2.  Ensure paths to individual model `evol_indices` files (within `EVE Ensemble/`) are correctly referenced or configured within `ensemble_evol_indices_analysis.py`.
    3.  Provide ground truth mutation data if required by the script.
    4.  Run the analysis:
        ```bash
        python ensemble_evol_indices_analysis.py 
        ```
    5.  Generate plots using the output from the analysis:
        ```bash
        python visualize_ensemble_results.py --results_file path/to/ensemble_results.csv
        ```

### 4. ESM-1b Fine-tuning
1.  Prepare your MSA file (e.g., `your_msa.a3m` or `your_msa.fasta`).
2.  Run the fine-tuning script:
    ```bash
    python fine_tune_esm1b.py --msa_file path/to/your_msa.a3m_or_fasta \\
                              --output_dir ESM/ESM1b/my_finetuned_model \\
                              --model_name facebook/esm1b_t33_650M_UR50S \\
                              --epochs 5 \\
                              --batch_size 1 \\
                              --learning_rate 1e-5 \\
                              --num_frozen_layers 30 
                              ...
    ```
    The fine-tuned model and logs will be saved in the specified output directory.

### 5. Sparse Autoencoder (SAE) Analysis
1.  Activate the `sae` conda environment: `conda activate sae`.
2.  Ensure ESM2 and SAE models are downloaded and correctly placed in `SAE/models/` as per `SAE/README.md`.
3.  Navigate to the `SAE/` directory.
4.  Prepare your input data: `mutated_sequences_with_scores.csv` (containing sequence variants and their EVE scores or other pathogenicity labels).
5.  Run the SAE probing script:
    ```bash
    python probe_sae.py 
    ```
    This will generate files like `results/weights/sae_raw_layer24.csv`.
6.  Run the visualization script:
    ```bash
    python visualize_sae.py
    ```
    This generates `highlight_units_layer24.pml` for PyMOL and `highlighted_latents_layer24.txt`. 