# MSA Data Collection and Filtering

This folder contains scripts and files related to the generation, filtering, and usage of multiple sequence alignments (MSAs) for the OLF domain.

## Step 1: Get MSA from hmmer

### Note: we provide the finalized dataset at `subsampled_MSA.a2m`. These steps are only to reproduce our results.

Ensure you have the [EVcouplings](https://github.com/debbiemarkslab/EVcouplings) repository cloned into this directory, as we use it for running hmmer.

```bash
git clone https://github.com/debbiemarkslab/EVcouplings.git
```

Ensure that you have [hmmer](http://hmmer.org/download.html) installed and available at the folder: `profilehmm/hmmer-3.4`. You also have to download the UniRef100 database and place it at `profilehmm/hmmer-3.4/bin/uniref100.fasta`.

You also need to create a conda environment using the `msagen.yml` file provided:

```bash
conda env create -f msagen.yml
conda activate msagen
```

For an example of how to run `hmmer` through EVcouplings, refer to `scripts/msagen.py`.

The MSA generated from this is located at `results/OLF_HUMAN/align/OLF_HUMAN.a2m`.

## Step 2: First round of filtering

Run `filtering.py` to remove short/duplicate sequences and filter the MSA to only include sequences with 200 residues or more. This generates a filtered MSA at `OLF_filtered.a2m`.

## Step 3: MMseqs2 clustering and subsampling

We use [MMseqs2](https://github.com/soedinglab/MMseqs2) to cluster the MSA.

(Tom add here your steps and python files, git clone, conda env, etc.)

This generates the subsampled MSA at `subsampled_MSA.a2m`.

## File Descriptions

| File/Folder                          | Description                                                                 |
|-------------------------------------|-----------------------------------------------------------------------------|
| `fasta/OLF.fasta`                   | FASTA file for the OLF domain.                                              |
| `scripts/msagen.py`                 | Script to run `hmmer` through EVcouplings.                                  |
| `scripts/config_files/OLF_HUMAN.txt`| Configuration file specifying `hmmer` parameters.                           |
| `results/OLF_HUMAN/align/OLF_HUMAN.a2m` | Output MSA from EVcouplings.                                            |
| `filtering.py`                      | First round of filtering: removes duplicate sequences and drops those shorter than 200 residues. |
| `OLF_filtered.a2m`                  | Filtered MSA after removing short/duplicate sequences.                      |
| `subsampled_MSA.a2m`                | Subsampled MSA after MMseqs2 clustering.                                    |

## For Phylogenetic Analysis

To see the sequences used for phylogeny, refer to:
```
../phylo/MSA_80cluster.fas
```
