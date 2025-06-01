"""
Visualize SAE embeddings of layer 24 of ESM2. Generates a PyMOL script and a text file with residue tags.
"""
import os
import sys
import torch
import numpy as np
import pandas as pd
from pathlib import Path
from safetensors.torch import load_file
from transformers import AutoTokenizer, AutoModelForMaskedLM
from scipy.signal import find_peaks
sys.path.append("..")
from interprot.interprot.sae_model import SparseAutoencoder
from interprot.interprot.utils import get_layer_activations

# ----------------------
# Configuration
# ----------------------
PLM_DIM = 1280
SAE_DIM = 4096
LAYER = 24
SEQUENCE = "GCGELVWVGEPLTLRTAETITGKYGVWMRDPKPTYPYTQETTWRIDTVGTDVRQVFEYDLISQFMQGYPSKVHILPRPLESTGAVVYSGSLYFQGAESRTVIRYELNTETVKAEKEIPGAGYHGQFPYSWGGYTDIDLAVDEAGLWVIYSTDEAKGAIVLSKLNPENLELEQTWETNIRKQSVANAFIICGTLYTVSSYTSADATVNFAYDTGTGISKTLTIPFKNRYKYSSMIDYNPLEKKLFAWDNLNMVTYDIKLSKM"

MODEL_DIR = Path("models")
CSV_PATH = Path("results/weights/sae_raw_layer24.csv")
SAE_PATH = Path("models/models--liambai--InterProt-ESM2-SAEs/snapshots/3ded2e51641df84d6b89f25430874e3a8eb42c24/esm2_plm1280_l24_sae4096.safetensors")
ESM_NAME = "facebook/esm2_t33_650M_UR50D"

OUTPUT_PML = "highlight_units_layer24.pml"
OUTPUT_TXT = "highlighted_latents_layer24.txt"

# ----------------------
# Load Models
# ----------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

esm_model = AutoModelForMaskedLM.from_pretrained(ESM_NAME, cache_dir=str(MODEL_DIR)).eval().to(device)
tokenizer = AutoTokenizer.from_pretrained(ESM_NAME)

sae_model = SparseAutoencoder(PLM_DIM, SAE_DIM).to(device).eval()
sae_model.load_state_dict(load_file(SAE_PATH), strict=True)

# ----------------------
# Extract Activations
# ----------------------
with torch.no_grad():
    layer_output = get_layer_activations(tokenizer, esm_model, [SEQUENCE], layer=LAYER)[0][1:-1].to(device)
    sae_activations = sae_model.get_acts(layer_output).cpu().numpy()

# ----------------------
# Identify Top Latents
# ----------------------
def extract_top_peaks(csv_path, activations, top_k=5):
    df = pd.read_csv(csv_path)
    sorted_df = df.sort_values("Weight", ascending=False)

    top_pos, top_neg = [], []

    for _, row in sorted_df.iterrows():
        idx = int(row["Index"])
        if row["Weight"] <= 0:
            break
        if np.any(activations[:, idx] != 0):
            top_pos.append(idx)
        if len(top_pos) == top_k:
            break

    for _, row in sorted_df[::-1].iterrows():
        idx = int(row["Index"])
        if row["Weight"] >= 0:
            break
        if np.any(activations[:, idx] != 0):
            top_neg.append(idx)
        if len(top_neg) == top_k:
            break

    unit_peaks = {}
    for idx in top_pos:
        peaks, _ = find_peaks(activations[:, idx], height=0)
        unit_peaks[(idx, "Positive")] = sorted((peaks + 1 + 243).tolist()) # Start from position 244 in accordance with the PDB file

    for idx in top_neg:
        peaks, _ = find_peaks(activations[:, idx], height=0)
        unit_peaks[(idx, "Negative")] = sorted((peaks + 1 + 243).tolist())

    return unit_peaks

unit_peaks = extract_top_peaks(CSV_PATH, sae_activations)

# ----------------------
# Generate PyMOL Script
# ----------------------
pml_lines = [
    "reinitialize",
    "fetch 4WXQ, async=0",
    "hide everything, 4WXQ",
    "show cartoon, 4WXQ",
    "set cartoon_smooth_loops, 1",
    "set cartoon_transparency, 0.1",
    "select keep_ions, (4WXQ and (resn NA+CA))",
    "show spheres, keep_ions",
    "hide labels, keep_ions",
    "select just_protein, 4WXQ and not keep_ions",
    "color slate, just_protein"
    "set sphere_scale, 0.4, keep_ions",
]

for (unit, sign), residues in unit_peaks.items():
    res_str = "+".join(map(str, residues))
    pml_lines += [
        f"\n# Layer {LAYER}, Unit {unit}, {sign}",
        f"select layer{LAYER}_unit_{unit}_residues, resi {res_str}",
        f"color red, layer{LAYER}_unit_{unit}_residues",
        "zoom complete",
        f"png layer{LAYER}_unit_{unit}.png, dpi=300, ray=1",
        f"save layer{LAYER}_unit_{unit}.pse",
        f"color lightblue, layer{LAYER}_unit_{unit}_residues",
        "deselect"
    ]

Path(OUTPUT_PML).write_text("\n".join(pml_lines))
print(f"Saved PyMOL script to: {OUTPUT_PML}")

# ----------------------
# Generate Residue Summary
# ----------------------
seq_offset = 1 + 243
resnum_to_aa = {r: SEQUENCE[r - 244] for r in range(244, 244 + len(SEQUENCE))}

with open(OUTPUT_TXT, "w") as f:
    f.write("# === Positive Latents ===\n")
    for (unit, sign), residues in unit_peaks.items():
        if sign.lower() != "positive":
            continue
        aa_tags = [f"{resnum_to_aa.get(r, 'X')}{r}" for r in residues]
        f.write(", ".join(aa_tags) + "\n")

    f.write("\n# === Negative Latents ===\n")
    for (unit, sign), residues in unit_peaks.items():
        if sign.lower() != "negative":
            continue
        aa_tags = [f"{resnum_to_aa.get(r, 'X')}{r}" for r in residues]
        f.write(", ".join(aa_tags) + "\n")

print(f"Saved latent-level highlights to: {OUTPUT_TXT}")
