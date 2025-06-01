#!/usr/bin/env python3
"""
Ensemble Analysis Script - Averaging Evolutionary Indices Before GMM

This script:
1. Runs GMM on individual model evol_indices (models 1-5)
2. For each ensemble size (2-5), averages evol_indices then runs GMM
3. Calculates accuracy against ground truth for each case
4. Plots accuracy vs ensemble size

Usage:
    python ensemble_evol_indices_analysis.py [--max_models 5] [--output_dir results]
"""

import os
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import argparse
import subprocess
from pathlib import Path
from sklearn.metrics import accuracy_score
import shutil

# Ground Truth Data
PATHOGENIC_MUTATIONS = [
    "C245Y", "G252R", "G246R", "R272G", "E323K", "G364V", "G367R", "P370L", "T377M", "T377K",
    "D380A", "V426F", "K423E", "C433R", "Y437H", "P481L", "I499F", "I499S", "N480K",
    "P481T", "S502P", "I477N", "I477S"
]

BENIGN_MUTATIONS = [
    "T293K", "V329M", "E352K", "K398R", "T353I", "K500R", "A445V"
]

def load_ground_truth():
    """Create ground truth DataFrame."""
    gt_data = []
    for mut in PATHOGENIC_MUTATIONS:
        gt_data.append({'mutation': mut, 'ground_truth': 'Pathogenic'})
    for mut in BENIGN_MUTATIONS:
        gt_data.append({'mutation': mut, 'ground_truth': 'Benign'})
    
    gt_df = pd.DataFrame(gt_data)
    print(f"Ground truth: {len(PATHOGENIC_MUTATIONS)} pathogenic, {len(BENIGN_MUTATIONS)} benign")
    return gt_df

def find_evol_indices_files(base_dir, max_models=5):
    """Find evolutionary indices files for each model."""
    evol_files = {}
    
    for i in range(1, max_models + 1):
        model_dir = Path(base_dir) / f"model_{i}" / "evol_indices"
        
        # Look for the evol_indices file
        evol_file = None
        if model_dir.exists():
            csv_files = list(model_dir.glob("*.csv"))
            if len(csv_files) == 1:
                evol_file = csv_files[0]
            elif len(csv_files) > 1:
                # Look for a specific pattern
                for f in csv_files:
                    if "2000" in f.name:
                        evol_file = f
                        break
                if evol_file is None:
                    evol_file = csv_files[0]
        
        if evol_file and evol_file.exists():
            evol_files[i] = evol_file
            print(f"Found evol indices for model {i}: {evol_file}")
        else:
            print(f"ERROR: Missing evol indices for model {i} in {model_dir}")
            return None
    
    if len(evol_files) != max_models:
        print(f"ERROR: Expected {max_models} evol indices files, found {len(evol_files)}")
        return None
    
    return evol_files

def average_evol_indices(evol_files, output_file):
    """Average evolutionary indices from multiple files."""
    print(f"Averaging evol indices from {len(evol_files)} files...")
    
    dfs = []
    for i, file_path in enumerate(evol_files):
        try:
            df = pd.read_csv(file_path)
            print(f"  Model {i+1}: {len(df)} mutations")
            
            # Check required columns
            if 'mutations' not in df.columns or 'evol_indices' not in df.columns:
                print(f"ERROR: Missing required columns in {file_path}")
                print(f"Available columns: {df.columns.tolist()}")
                return False
            
            # Keep only the columns we need
            df = df[['mutations', 'evol_indices']].copy()
            dfs.append(df.set_index('mutations'))
            
        except Exception as e:
            print(f"ERROR reading {file_path}: {e}")
            return False
    
    try:
        # Combine all dataframes
        combined = pd.concat(dfs, axis=1, sort=False)
        
        # Average the evol_indices columns
        averaged = combined['evol_indices'].mean(axis=1)
        
        # Create output dataframe with required columns
        result_df = averaged.reset_index()
        result_df.columns = ['mutations', 'evol_indices']
        result_df.insert(0, 'protein_name', 'OLF')  # Add protein_name column
        
        # Save to file
        result_df.to_csv(output_file, index=False)
        print(f"Saved averaged indices ({len(result_df)} mutations) to: {output_file}")
        return True
        
    except Exception as e:
        print(f"ERROR during averaging: {e}")
        return False

def setup_gmm_directories(output_dir, analysis_name):
    """Set up the directory structure needed for GMM analysis."""
    base_dir = Path(output_dir) / analysis_name
    
    # Create directory structure
    evol_indices_dir = base_dir / "evol_indices"
    eve_scores_dir = base_dir / "eve_scores"  
    gmm_models_dir = base_dir / "gmm_models"
    
    for dir_path in [evol_indices_dir, eve_scores_dir, gmm_models_dir]:
        dir_path.mkdir(parents=True, exist_ok=True)
    
    return base_dir, evol_indices_dir, eve_scores_dir, gmm_models_dir

def run_gmm_analysis(evol_indices_file, base_dir, analysis_name):
    """Run the GMM analysis on evol indices."""
    print(f"Running GMM analysis for {analysis_name}...")
    
    base_dir = Path(base_dir)
    evol_indices_dir = base_dir / "evol_indices"
    eve_scores_dir = base_dir / "eve_scores"
    gmm_models_dir = base_dir / "gmm_models"
    
    # Copy the evol indices to the expected location
    target_evol_file = evol_indices_dir / "OLF_2000_samples.csv"
    
    try:
        shutil.copy2(evol_indices_file, target_evol_file)
        print(f"Copied evol indices to: {target_evol_file}")
        
        # Verify the file exists and has content
        if not target_evol_file.exists():
            print(f"ERROR: Copied file does not exist: {target_evol_file}")
            return None
        
        file_size = target_evol_file.stat().st_size
        print(f"Evol indices file size: {file_size} bytes")
        
    except Exception as e:
        print(f"ERROR copying evol indices file: {e}")
        return None
    
    # Create protein list file
    protein_list_file = base_dir / "protein_list.csv"
    pd.DataFrame({'protein_name': ['OLF']}).to_csv(protein_list_file, index=False)
    print(f"Created protein list file: {protein_list_file}")
    
    # Clean output directories
    if eve_scores_dir.exists():
        shutil.rmtree(eve_scores_dir)
    if gmm_models_dir.exists():
        shutil.rmtree(gmm_models_dir)
    
    # Recreate directories
    eve_scores_dir.mkdir(parents=True, exist_ok=True)
    gmm_models_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Cleaned and recreated output directories")
    
    # Prepare GMM command
    cmd = [
        "python", "./train_GMM_and_compute_EVE_scores.py",
        "--input_evol_indices_location", str(evol_indices_dir),
        "--input_evol_indices_filename_suffix", "_2000_samples",
        "--protein_list", str(protein_list_file),
        "--output_eve_scores_location", str(eve_scores_dir),
        "--output_eve_scores_filename_suffix", f"_{analysis_name}",
        "--GMM_parameter_location", str(gmm_models_dir),
        "--protein_GMM_weight", "0.3",
        "--compute_EVE_scores",
        "--verbose"
    ]
    
    print(f"Running GMM command for {analysis_name}...")
    
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
        
        print(f"GMM command completed with return code: {result.returncode}")
        
        if result.returncode == 0:
            print(f"GMM command succeeded for {analysis_name}")
            
            # Look for the expected EVE scores file
            eve_files = list(eve_scores_dir.glob(f"*{analysis_name}*.csv"))
            
            if eve_files:
                eve_file = eve_files[0]
                print(f"Found EVE scores file: {eve_file} ({eve_file.stat().st_size} bytes)")
                return eve_file
            else:
                print(f"ERROR: No EVE scores file found for {analysis_name}")
                return None
        else:
            print(f"ERROR: GMM analysis failed for {analysis_name}")
            if result.stderr:
                print("STDERR:", result.stderr)
            return None
            
    except subprocess.TimeoutExpired:
        print(f"ERROR: GMM analysis timed out for {analysis_name}")
        return None
    except Exception as e:
        print(f"ERROR running GMM analysis: {e}")
        return None

def calculate_accuracy(eve_scores_file, ground_truth_df, analysis_name):
    """Calculate accuracy using EVE_classes_100_pct_retained."""
    print(f"Calculating accuracy for {analysis_name} from: {eve_scores_file}")
    
    try:
        df = pd.read_csv(eve_scores_file)
        print(f"Loaded {len(df)} predictions")
        
        # Check for required columns
        if 'mutations' not in df.columns:
            print("ERROR: 'mutations' column not found")
            return None
        
        if 'EVE_classes_100_pct_retained' not in df.columns:
            print("ERROR: 'EVE_classes_100_pct_retained' column not found")
            print(f"Available columns: {df.columns.tolist()}")
            return None
        
        # Merge with ground truth
        merged = pd.merge(ground_truth_df, df[['mutations', 'EVE_classes_100_pct_retained']], 
                         left_on='mutation', right_on='mutations', how='inner')
        
        if len(merged) == 0:
            print("ERROR: No overlap between ground truth and predictions")
            return None
        
        print(f"Found {len(merged)} overlapping mutations")
        
        # Calculate accuracy
        correct = (merged['ground_truth'] == merged['EVE_classes_100_pct_retained']).sum()
        total = len(merged)
        accuracy = correct / total
        
        print(f"Accuracy: {correct}/{total} = {accuracy:.3f} ({accuracy*100:.1f}%)")
        
        # Show breakdown
        path_mask = merged['ground_truth'] == 'Pathogenic'
        benign_mask = merged['ground_truth'] == 'Benign'
        
        path_correct = (merged[path_mask]['ground_truth'] == merged[path_mask]['EVE_classes_100_pct_retained']).sum()
        path_total = path_mask.sum()
        
        benign_correct = (merged[benign_mask]['ground_truth'] == merged[benign_mask]['EVE_classes_100_pct_retained']).sum()
        benign_total = benign_mask.sum()
        
        if path_total > 0:
            print(f"  Pathogenic: {path_correct}/{path_total} = {path_correct/path_total:.3f}")
        if benign_total > 0:
            print(f"  Benign: {benign_correct}/{benign_total} = {benign_correct/benign_total:.3f}")
        
        return accuracy
        
    except Exception as e:
        print(f"ERROR calculating accuracy: {e}")
        return None

def plot_results(results, output_file):
    """Plot accuracy vs ensemble configuration."""
    plt.figure(figsize=(12, 8))
    
    # Separate individual models from ensemble results
    individual_results = {k: v for k, v in results.items() if k.startswith('model_')}
    ensemble_results = {k: v for k, v in results.items() if k.startswith('ensemble_')}
    
    # Plot individual models
    if individual_results:
        models = sorted([int(k.split('_')[1]) for k in individual_results.keys()])
        individual_accs = [individual_results[f'model_{m}'] for m in models]
        
        plt.subplot(2, 1, 1)
        plt.bar(models, individual_accs, alpha=0.7, color='skyblue', label='Individual Models')
        plt.title('Individual Model Accuracies', fontsize=14)
        plt.xlabel('Model Number', fontsize=12)
        plt.ylabel('Accuracy', fontsize=12)
        plt.ylim(0, 1)
        plt.grid(True, alpha=0.3)
        
        # Add value labels
        for i, acc in enumerate(individual_accs):
            plt.annotate(f'{acc:.3f}', (models[i], acc), ha='center', va='bottom')
    
    # Plot ensemble results
    if ensemble_results:
        ensemble_sizes = sorted([int(k.split('_')[1]) for k in ensemble_results.keys()])
        ensemble_accs = [ensemble_results[f'ensemble_{s}'] for s in ensemble_sizes]
        
        plt.subplot(2, 1, 2)
        plt.plot(ensemble_sizes, ensemble_accs, 'ro-', linewidth=2, markersize=8, label='Ensemble')
        plt.title('Ensemble Accuracy vs Number of Models', fontsize=14)
        plt.xlabel('Number of Models in Ensemble', fontsize=12)
        plt.ylabel('Accuracy', fontsize=12)
        plt.grid(True, alpha=0.3)
        plt.xticks(ensemble_sizes)
        plt.ylim(0, 1)
        
        # Add value labels
        for size, acc in zip(ensemble_sizes, ensemble_accs):
            plt.annotate(f'{acc:.3f}', (size, acc), textcoords="offset points", 
                        xytext=(0,10), ha='center', fontsize=10)
    
    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved plot to: {output_file}")

def main():
    parser = argparse.ArgumentParser(description='Analyze ensemble performance by averaging evol indices')
    parser.add_argument('--base_dir', type=str, default='.',
                        help='Base directory containing model_1, model_2, etc.')
    parser.add_argument('--output_dir', type=str, default='ensemble_results',
                        help='Output directory for results')
    parser.add_argument('--max_models', type=int, default=5,
                        help='Maximum number of models to analyze')
    parser.add_argument('--skip_individual', action='store_true',
                        help='Skip individual model analysis')
    parser.add_argument('--skip_ensemble', action='store_true',
                        help='Skip ensemble analysis')
    
    args = parser.parse_args()
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True)
    
    print("="*80)
    print("ENSEMBLE EVOL INDICES ANALYSIS")
    print("="*80)
    
    # Load ground truth
    ground_truth_df = load_ground_truth()
    
    # Find evolutionary indices files
    evol_files = find_evol_indices_files(args.base_dir, args.max_models)
    if evol_files is None:
        sys.exit(1)
    
    # Results storage
    results = {}
    
    # 1. Analyze individual models
    if not args.skip_individual:
        print(f"\n{'='*20} INDIVIDUAL MODEL ANALYSIS {'='*20}")
        
        for model_num in range(1, args.max_models + 1):
            print(f"\n--- Processing Model {model_num} ---")
            
            # Set up directories
            analysis_name = f"model_{model_num}"
            base_dir, evol_dir, scores_dir, gmm_dir = setup_gmm_directories(output_dir, analysis_name)
            
            # Run GMM analysis on individual model
            eve_scores_file = run_gmm_analysis(evol_files[model_num], base_dir, analysis_name)
            if eve_scores_file is None:
                print(f"SKIPPING model {model_num} due to GMM error")
                continue
            
            # Calculate accuracy
            accuracy = calculate_accuracy(eve_scores_file, ground_truth_df, analysis_name)
            if accuracy is not None:
                results[analysis_name] = accuracy
                print(f"RESULT: Model {model_num} → Accuracy = {accuracy:.3f}")
    
    # 2. Analyze ensemble sizes
    if not args.skip_ensemble:
        print(f"\n{'='*20} ENSEMBLE ANALYSIS {'='*20}")
        
        for ensemble_size in range(2, args.max_models + 1):
            print(f"\n--- Processing Ensemble Size {ensemble_size} ---")
            
            # Set up directories
            analysis_name = f"ensemble_{ensemble_size}"
            base_dir, evol_dir, scores_dir, gmm_dir = setup_gmm_directories(output_dir, analysis_name)
            
            # Average evol indices for this ensemble size
            evol_files_subset = [evol_files[i] for i in range(1, ensemble_size + 1)]
            averaged_indices_file = base_dir / f"averaged_evol_indices_n{ensemble_size}.csv"
            
            success = average_evol_indices(evol_files_subset, averaged_indices_file)
            if not success:
                print(f"SKIPPING ensemble size {ensemble_size} due to averaging error")
                continue
            
            # Run GMM analysis on averaged indices
            eve_scores_file = run_gmm_analysis(averaged_indices_file, base_dir, analysis_name)
            if eve_scores_file is None:
                print(f"SKIPPING ensemble size {ensemble_size} due to GMM error")
                continue
            
            # Calculate accuracy
            accuracy = calculate_accuracy(eve_scores_file, ground_truth_df, analysis_name)
            if accuracy is not None:
                results[analysis_name] = accuracy
                print(f"RESULT: Ensemble Size {ensemble_size} → Accuracy = {accuracy:.3f}")
    
    # Generate final results
    if results:
        print(f"\n{'='*20} FINAL RESULTS {'='*20}")
        
        # Individual model results
        individual_results = {k: v for k, v in results.items() if k.startswith('model_')}
        if individual_results:
            print("\nIndividual Model Accuracies:")
            for model_name in sorted(individual_results.keys()):
                acc = individual_results[model_name]
                print(f"  {model_name}: {acc:.3f} ({acc*100:.1f}%)")
        
        # Ensemble results
        ensemble_results = {k: v for k, v in results.items() if k.startswith('ensemble_')}
        if ensemble_results:
            print("\nEnsemble Accuracies:")
            for ensemble_name in sorted(ensemble_results.keys()):
                acc = ensemble_results[ensemble_name]
                print(f"  {ensemble_name}: {acc:.3f} ({acc*100:.1f}%)")
        
        # Save results to CSV
        results_df = pd.DataFrame([
            {'analysis': k, 'accuracy': v} for k, v in results.items()
        ])
        results_file = output_dir / "ensemble_analysis_results.csv"
        results_df.to_csv(results_file, index=False)
        print(f"\nSaved results to: {results_file}")
        
        # Generate plot
        plot_file = output_dir / "ensemble_analysis_plot.png"
        plot_results(results, plot_file)
        
        # Show improvement if we have ensemble results
        if len(ensemble_results) > 1:
            ensemble_accs = [ensemble_results[f'ensemble_{s}'] for s in sorted([int(k.split('_')[1]) for k in ensemble_results.keys()])]
            improvement = ensemble_accs[-1] - ensemble_accs[0]
            print(f"\nEnsemble improvement: {improvement:.3f} ({improvement*100:.1f} percentage points)")
    
    else:
        print("\nERROR: No successful analyses completed")
        sys.exit(1)
    
    print(f"\nAnalysis complete! Results saved in: {output_dir}")

if __name__ == '__main__':
    main() 