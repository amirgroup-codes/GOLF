#!/usr/bin/env python3
"""
Filter MSA A2M File for Redundant or Low-Quality Sequences

This script reads an A2M-formatted alignment file, removes duplicate sequences,
removes sequences shorter than 200 residues, and sequences with <200 amino acids
(excluding '.' and '-'). The result is saved as a filtered A2M file.
"""

import argparse
import pandas as pd
from pathlib import Path


def parse_a2m(file_path):
    """Parse A2M alignment file into a pandas DataFrame"""
    ids, seqs = [], []
    with open(file_path, 'r') as f:
        seq = []
        for line in f:
            line = line.strip()
            if line.startswith('>'):
                if seq:
                    seqs.append(''.join(seq))
                    seq = []
                ids.append(line[1:])
            else:
                seq.append(line)
        if seq:
            seqs.append(''.join(seq))
    return pd.DataFrame({'ID': ids, 'Sequence': seqs})


def main(args):
    msa = parse_a2m(args.input)
    print(f"Original MSA size: {msa.shape[0]}")

    msa = msa.drop_duplicates(subset="Sequence", keep="first")
    print(f"After duplicate removal: {msa.shape[0]}")

    msa["Length_NoSpecialChars"] = msa["Sequence"].apply(lambda s: len(s.replace('.', '').replace('-', '')))
    msa = msa[msa["Length_NoSpecialChars"] >= 200].copy()
    print(f"After filtering sequences with <200 valid AAs: {msa.shape[0]}")

    output_path = Path(args.output)
    with open(output_path, 'w') as f:
        for _, row in msa.iterrows():
            f.write(f">{row['ID']}\n{row['Sequence']}\n")

    print(f"Filtered A2M saved to: {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Filter A2M file by removing duplicates and short/low-quality sequences")
    parser.add_argument("--input", type=str, default="results/OLF_HUMAN/align/OLF_HUMAN.a2m", help="Path to input .a2m file")
    parser.add_argument("--output", type=str, default="OLF_filtered.a2m", help="Path to output filtered A2M file")
    args = parser.parse_args()
    main(args)
