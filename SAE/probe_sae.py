"""
Probe the SAE embeddings of layer 24 of ESM2
"""
import numpy as np
import pandas as pd
import torch
from pathlib import Path
import argparse
import datetime
from sklearn.linear_model import Ridge
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.metrics import make_scorer, r2_score
from transformers import AutoTokenizer, EsmModel
from safetensors.torch import load_file
from tqdm import tqdm
from scipy.stats import spearmanr
import contextlib
from torch.cuda.amp import autocast as cuda_autocast
import sys
sys.path.append("..")
from interprot.interprot.sae_model import SparseAutoencoder
from interprot.interprot.utils import get_layer_activations

# ----------------------------
# Argument Parsing
# ----------------------------
parser = argparse.ArgumentParser(description='Probe SAE embeddings only')
parser.add_argument('--output_dir', type=str, default='results', help='Output directory')
parser.add_argument('--batch_size', type=int, default=64, help='Batch size')
parser.add_argument('--random_state', type=int, default=0, help='Random seed')
args = parser.parse_args()

# ----------------------------
# Configuration
# ----------------------------
PLM_DIM = 1280
SAE_DIM = 4096
LAYER = 24
MODEL_DIR = Path("models")
ESM_NAME = "facebook/esm2_t33_650M_UR50D"
SAE_CHECKPOINT = MODEL_DIR / "models--liambai--InterProt-ESM2-SAEs/snapshots/3ded2e51641df84d6b89f25430874e3a8eb42c24/esm2_plm1280_l24_sae4096.safetensors"
MODEL_DIR.mkdir(parents=True, exist_ok=True)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ----------------------------
# Output Setup
# ----------------------------
results_dir = Path(args.output_dir)
weights_dir = results_dir / "weights"
weights_dir.mkdir(parents=True, exist_ok=True)

# ----------------------------
# Load Input Data
# ----------------------------
df = pd.read_csv("mutated_sequences_with_scores.csv")
sequences = df["full_sequence"].tolist()
scores = df["EVE_score"].tolist()

# ----------------------------
# Load Models
# ----------------------------
print("Loading models...")
tokenizer = AutoTokenizer.from_pretrained(ESM_NAME, cache_dir=str(MODEL_DIR))
esm_model = EsmModel.from_pretrained(ESM_NAME, cache_dir=str(MODEL_DIR)).to(device).eval()
sae_model = SparseAutoencoder(PLM_DIM, SAE_DIM).to(device).eval()
sae_model.load_state_dict(load_file(str(SAE_CHECKPOINT)), strict=True)

# ----------------------------
# Compute SAE Embeddings
# ----------------------------
print("Extracting embeddings...")
sae_acts_list = []

for start in tqdm(range(0, len(sequences), args.batch_size), desc="Embedding"):
    batch = sequences[start:start + args.batch_size]
    with torch.inference_mode(), (cuda_autocast(dtype=torch.float16) if device.type == "cuda" else contextlib.nullcontext()):
        activations = get_layer_activations(tokenizer, esm_model, batch, layer=LAYER, device=device)

    for act in activations:
        core = act[1:-1]  # remove [CLS] and [EOS]
        if core.numel() == 0:
            sae_acts_list.append(torch.zeros(SAE_DIM).cpu().numpy())
        else:
            sae_vector = sae_model.get_acts(core)
            sae_acts_list.append(torch.mean(sae_vector, dim=0).cpu().numpy())

sae_acts = np.asarray(sae_acts_list)

# ----------------------------
# Train Linear Probe
# ----------------------------
print("Training probe...")
X_train, X_test, y_train, y_test = train_test_split(sae_acts, scores, test_size=0.2, random_state=args.random_state)

alphas = [0.001, 0.01, 0.1, 1, 10, 100]
scorer = make_scorer(lambda y_true, y_pred: spearmanr(y_true, y_pred)[0])

grid = GridSearchCV(Ridge(), {'alpha': alphas}, cv=5, scoring=scorer)
grid.fit(X_train, y_train)
best_model = grid.best_estimator_

# ----------------------------
# Evaluate
# ----------------------------
y_pred = best_model.predict(X_test)
r2 = r2_score(y_test, y_pred)
spearman = spearmanr(y_test, y_pred)[0]
print(f"R² = {r2:.4f}, Spearman = {spearman:.4f}")

# ----------------------------
# Save Probe Weights
# ----------------------------
coef = best_model.coef_
sorted_weights = sorted(enumerate(coef), key=lambda x: abs(x[1]), reverse=True)
pd.DataFrame(sorted_weights, columns=["Index", "Weight"]).to_csv(weights_dir / f"sae_raw_layer{LAYER}.csv", index=False)
print(f"Saved probe weights to {weights_dir / f'sae_raw_layer{LAYER}.csv'}")
