#!/usr/bin/env python3
"""
Visualize GMM Decision Boundary for Benign vs Pathogenic Classification

This script reads GMM parameters from the ensemble analysis results and creates
a visualization showing:
1. Benign Gaussian distribution (blue)
2. Pathogenic Gaussian distribution (red) 
3. Decision boundary where classifications change
4. Optional: overlay of actual data points if available
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import argparse
from pathlib import Path
from scipy.stats import norm
from scipy.optimize import fsolve
import seaborn as sns

# Set style for publication-quality plots
plt.style.use('seaborn-v0_8-darkgrid')

def find_gmm_stats_file(analysis_dir):
    """Find the GMM stats file from ensemble analysis."""
    analysis_path = Path(analysis_dir)
    
    # Look for GMM stats files in ensemble directories
    possible_patterns = [
        "**/GMM_stats*.csv",
        "**/gmm_models/**/GMM_stats*.csv",
        "ensemble_*/gmm_models/*/GMM_stats*.csv"
    ]
    
    for pattern in possible_patterns:
        files = list(analysis_path.glob(pattern))
        if files:
            # Use the most recent ensemble (highest number)
            files.sort(key=lambda x: str(x))
            gmm_file = files[-1]
            print(f"Found GMM stats file: {gmm_file}")
            return gmm_file
    
    return None

def load_gmm_parameters(gmm_file):
    """Load GMM parameters from the stats file."""
    try:
        df = pd.read_csv(gmm_file)
        
        # Look for OLF protein or use the last row
        if 'protein_name' in df.columns:
            olf_row = df[df['protein_name'] == 'OLF']
            if not olf_row.empty:
                params = olf_row.iloc[0]
            else:
                params = df.iloc[-1]  # Use last row if OLF not found
        else:
            params = df.iloc[-1]
        
        gmm_params = {
            'weight_pathogenic': float(params['weight_pathogenic']),
            'mean_pathogenic': float(params['mean_pathogenic']),
            'mean_benign': float(params['mean_benign']),
            'std_pathogenic': float(params['std_dev_pathogenic']),
            'std_benign': float(params['std_dev_benign'])
        }
        
        gmm_params['weight_benign'] = 1.0 - gmm_params['weight_pathogenic']
        
        print("GMM Parameters:")
        print(f"  Pathogenic: mean={gmm_params['mean_pathogenic']:.3f}, std={gmm_params['std_pathogenic']:.3f}, weight={gmm_params['weight_pathogenic']:.3f}")
        print(f"  Benign: mean={gmm_params['mean_benign']:.3f}, std={gmm_params['std_benign']:.3f}, weight={gmm_params['weight_benign']:.3f}")
        
        return gmm_params
        
    except Exception as e:
        print(f"Error loading GMM parameters: {e}")
        return None

def simulate_gmm_parameters():
    """Create realistic GMM parameters for demonstration."""
    print("Using simulated GMM parameters for demonstration...")
    
    # Realistic parameters based on typical EVE results
    params = {
        'weight_pathogenic': 0.65,
        'mean_pathogenic': 6.1,
        'mean_benign': 2.6,
        'std_pathogenic': 1.45,
        'std_benign': 2.15,
        'weight_benign': 0.35
    }
    
    print("Simulated GMM Parameters:")
    print(f"  Pathogenic: mean={params['mean_pathogenic']:.3f}, std={params['std_pathogenic']:.3f}, weight={params['weight_pathogenic']:.3f}")
    print(f"  Benign: mean={params['mean_benign']:.3f}, std={params['std_benign']:.3f}, weight={params['weight_benign']:.3f}")
    
    return params

def find_decision_boundary(params):
    """Find the decision boundary where P(pathogenic) = P(benign)."""
    
    def gmm_difference(x):
        """Function to find where pathogenic and benign probabilities are equal."""
        p_pathogenic = params['weight_pathogenic'] * norm.pdf(x, params['mean_pathogenic'], params['std_pathogenic'])
        p_benign = params['weight_benign'] * norm.pdf(x, params['mean_benign'], params['std_benign'])
        return p_pathogenic - p_benign
    
    # Find intersection points
    try:
        # Search in reasonable range around the means
        search_range = np.linspace(
            min(params['mean_benign'] - 3*params['std_benign'], 
                params['mean_pathogenic'] - 3*params['std_pathogenic']),
            max(params['mean_benign'] + 3*params['std_benign'], 
                params['mean_pathogenic'] + 3*params['std_pathogenic']),
            1000
        )
        
        # Find sign changes (intersections)
        differences = [gmm_difference(x) for x in search_range]
        boundaries = []
        
        for i in range(len(differences)-1):
            if differences[i] * differences[i+1] < 0:  # Sign change
                # Refine the boundary location
                boundary = fsolve(gmm_difference, search_range[i])[0]
                boundaries.append(boundary)
        
        print(f"Found {len(boundaries)} decision boundaries: {[f'{b:.3f}' for b in boundaries]}")
        return boundaries
        
    except Exception as e:
        print(f"Error finding decision boundary: {e}")
        return []

def load_evol_indices_data(analysis_dir):
    """Try to load actual evolutionary indices data for overlay."""
    analysis_path = Path(analysis_dir)
    
    # Look for evol indices files
    possible_files = [
        "**/eve_scores/all_EVE_scores*.csv",
        "**/*evol_indices*.csv",
        "**/evol_indices/*.csv"
    ]
    
    for pattern in possible_files:
        files = list(analysis_path.glob(pattern))
        if files:
            try:
                # Use the most recent file
                files.sort(key=lambda x: str(x))
                data_file = files[-1]
                df = pd.read_csv(data_file)
                
                if 'evol_indices' in df.columns:
                    print(f"Loaded evolutionary indices data from: {data_file}")
                    return df['evol_indices'].values
                elif 'EVE_scores' in df.columns:
                    print(f"Loaded EVE scores data from: {data_file}")
                    return df['EVE_scores'].values
                    
            except Exception as e:
                print(f"Error loading data from {data_file}: {e}")
                continue
    
    print("No evolutionary indices data found for overlay")
    return None

def load_eve_scores_with_classifications(analysis_dir):
    """Load EVE scores data with classifications for histogram coloring."""

    # Helper function to process a DataFrame from a loaded CSV file
    def _process_df(df_loaded, file_path_str):
        required_classification_col = 'EVE_classes_100_pct_retained'
        mutations_col_name = 'mutations' # Standard name for the mutations column

        if required_classification_col not in df_loaded.columns:
            print(f"    Column '{required_classification_col}' not found in {Path(file_path_str).name}.")
            return None

        # Initialize a dictionary to build the new DataFrame an`d ensure column uniqueness
        data_for_processed_df = {}

        # Add classification column
        data_for_processed_df[required_classification_col] = df_loaded[required_classification_col].copy()

        # Add mutations column if it exists
        if mutations_col_name in df_loaded.columns:
            data_for_processed_df[mutations_col_name] = df_loaded[mutations_col_name].copy()

        # 1. Determine and add scores for GMM plots (output column name: 'EVE_scores')
        # This data is used by histogram plots that overlay GMM parameters.
        if 'evol_indices' in df_loaded.columns:
            print(f"    Found 'evol_indices' column. Using its data for GMM plots (will be named 'EVE_scores' in output DataFrame).")
            data_for_processed_df['EVE_scores'] = df_loaded['evol_indices'].copy()
        elif 'EVE_scores' in df_loaded.columns: # This is the literal 'EVE_scores' from CSV as fallback for GMM
            print(f"    'evol_indices' not found. Using data from CSV's 'EVE_scores' column for GMM plots (will be named 'EVE_scores' in output DataFrame).")
            data_for_processed_df['EVE_scores'] = df_loaded['EVE_scores'].copy()
        else:
            print(f"    CRITICAL: Neither 'evol_indices' nor 'EVE_scores' (from CSV) found. Cannot provide scores for GMM plots.")
            return None

        # 2. Determine and add scores for the Box Plot (output column name: 'EVE_scores_for_boxplot')
        # This MUST come from a column literally named 'EVE_scores' in the original CSV, if it exists.
        if 'EVE_scores' in df_loaded.columns: # Checking for the literal 'EVE_scores' column in the input CSV
            print(f"    Found literal 'EVE_scores' column in CSV. Using its data for the Box Plot (will be named 'EVE_scores_for_boxplot' in output DataFrame).")
            data_for_processed_df['EVE_scores_for_boxplot'] = df_loaded['EVE_scores'].copy()
        else:
            print(f"    Literal 'EVE_scores' column not found in CSV. Box plot will need to use GMM scores ('{data_for_processed_df.get('EVE_scores', {}).name}') as fallback.")
            # 'EVE_scores_for_boxplot' will not be added to data_for_processed_df if the literal column doesn't exist.

        processed_df = pd.DataFrame(data_for_processed_df)

        # Verify critical GMM scores column exists
        if 'EVE_scores' not in processed_df.columns:
             print("CRITICAL ERROR INTERNAL: 'EVE_scores' column for GMM is missing after processing. This should not happen.")
             return None

        class_counts = processed_df[required_classification_col].value_counts()
        print(f"    Total sequences from file: {len(processed_df)}")
        print(f"    Classifications found: {dict(class_counts)}")
        return processed_df

    # --- Main logic for load_eve_scores_with_classifications ---
    base_path = Path(analysis_dir) # analysis_dir comes from args.analysis_dir (defaults to '.')
    
    # Attempt 1: User-specified hardcoded path, interpreted relative to the script's CWD.
    # Path should be: ./Ensemble_Analysis_Fixed/ensemble_5/eve_scores/
    hardcoded_eve_data_dir = Path("Ensemble_Analysis_Fixed/ensemble_5/eve_scores")

    print(f"Looking for EVE scores data...")
    if hardcoded_eve_data_dir.is_dir():
        print(f"Attempting to load EVE scores from specified directory: {hardcoded_eve_data_dir}")
        # Look for any CSV file, prioritizing newer ones if multiple exist.
        csv_files = sorted(list(hardcoded_eve_data_dir.glob("*.csv")), key=lambda p: p.stat().st_mtime, reverse=True)
        
        if not csv_files:
            print(f"  No CSV files found in {hardcoded_eve_data_dir}.")

        for data_file in csv_files:
            print(f"  Trying file: {data_file.name}")
            try:
                df_loaded = pd.read_csv(data_file)
                processed_df = _process_df(df_loaded, str(data_file))
                if processed_df is not None:
                    return processed_df
            except Exception as e:
                print(f"    Error loading or processing {data_file.name}: {e}")
                continue 
    else:
        print(f"Specified directory not found or is not a directory: {hardcoded_eve_data_dir}")

    print(f"Could not find a suitable EVE scores file in the specified directory. Falling back to general search patterns in '{analysis_dir}'...")
    
    # Attempt 2: Original glob pattern search (fallback, relative to analysis_dir)
    possible_patterns = [
        "**/eve_scores/all_EVE_scores*.csv",
        "**/all_EVE_scores*.csv",
        "**/*EVE_scores*.csv"
    ]
    
    for pattern in possible_patterns:
        # Search relative to analysis_dir (which is base_path)
        files_found = sorted(list(base_path.glob(pattern)), key=lambda p: p.stat().st_mtime, reverse=True)
        
        if files_found:
            # print(f"Found {len(files_found)} potential file(s) with pattern '{pattern}'. First one: {files_found[0]}")
            pass # Reduce verbosity unless a file is actually processed

        for data_file_path in files_found: # Corrected variable name
            print(f"  Trying file (glob pattern '{pattern}'): {data_file_path}")
            try:
                df_loaded = pd.read_csv(data_file_path)
                processed_df = _process_df(df_loaded, str(data_file_path))
                if processed_df is not None:
                    return processed_df
            except Exception as e:
                print(f"    Error loading or processing {data_file_path}: {e}") # Corrected variable name
                continue
    
    print("No EVE scores with classifications found after all attempts.")
    return None

def create_gmm_decision_boundary_plot(params, output_file, evol_data=None):
    """Create the main GMM decision boundary visualization."""
    
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Define the range for plotting
    x_min = min(params['mean_benign'] - 4*params['std_benign'], 
                params['mean_pathogenic'] - 4*params['std_pathogenic'])
    x_max = max(params['mean_benign'] + 4*params['std_benign'], 
                params['mean_pathogenic'] + 4*params['std_pathogenic'])
    
    x = np.linspace(x_min, x_max, 1000)
    
    # Calculate the probability densities
    p_pathogenic = params['weight_pathogenic'] * norm.pdf(x, params['mean_pathogenic'], params['std_pathogenic'])
    p_benign = params['weight_benign'] * norm.pdf(x, params['mean_benign'], params['std_benign'])
    
    # Plot filled curves
    ax.fill_between(x, 0, p_benign, alpha=0.6, color='dodgerblue', 
                   label=f'Benign (μ={params["mean_benign"]:.2f}, σ={params["std_benign"]:.2f})')
    ax.fill_between(x, 0, p_pathogenic, alpha=0.6, color='crimson', 
                   label=f'Pathogenic (μ={params["mean_pathogenic"]:.2f}, σ={params["std_pathogenic"]:.2f})')
    
    # Plot the curve outlines for clarity
    ax.plot(x, p_benign, color='darkblue', linewidth=2, alpha=0.8)
    ax.plot(x, p_pathogenic, color='darkred', linewidth=2, alpha=0.8)
    
    # Add data points overlay if available
    if evol_data is not None and len(evol_data) > 0:
        # Create histogram of actual data
        hist, bins = np.histogram(evol_data, bins=30, density=True)
        bin_centers = (bins[:-1] + bins[1:]) / 2
        ax.plot(bin_centers, hist * 0.1, 'k-', alpha=0.5, linewidth=1, label='Actual Data (scaled)')
    
    # Styling
    ax.set_title('GMM Decision Boundary: Benign vs Pathogenic Classification', 
                fontsize=16, fontweight='bold', pad=20)
    ax.set_xlabel('Evolutionary Index', fontsize=14)
    ax.set_ylabel('Probability Density', fontsize=14)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=12, loc='upper right')
    ax.set_ylim(bottom=0)
    
    plt.tight_layout()
    
    # Save as PDF if filename ends with .pdf, otherwise PNG
    if str(output_file).endswith('.pdf'):
        plt.savefig(output_file, format='pdf', dpi=300, bbox_inches='tight')
    else:
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved GMM decision boundary plot to: {output_file}")

def create_gmm_decision_boundary_with_means_plot(params, output_file, evol_data=None):
    """Create the GMM decision boundary visualization with mean lines."""
    
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Define the range for plotting
    x_min = min(params['mean_benign'] - 4*params['std_benign'], 
                params['mean_pathogenic'] - 4*params['std_pathogenic'])
    x_max = max(params['mean_benign'] + 4*params['std_benign'], 
                params['mean_pathogenic'] + 4*params['std_pathogenic'])
    
    x = np.linspace(x_min, x_max, 1000)
    
    # Calculate the probability densities
    p_pathogenic = params['weight_pathogenic'] * norm.pdf(x, params['mean_pathogenic'], params['std_pathogenic'])
    p_benign = params['weight_benign'] * norm.pdf(x, params['mean_benign'], params['std_benign'])
    
    # Plot filled curves
    ax.fill_between(x, 0, p_benign, alpha=0.6, color='dodgerblue', 
                   label=f'Benign (μ={params["mean_benign"]:.2f}, σ={params["std_benign"]:.2f})')
    ax.fill_between(x, 0, p_pathogenic, alpha=0.6, color='crimson', 
                   label=f'Pathogenic (μ={params["mean_pathogenic"]:.2f}, σ={params["std_pathogenic"]:.2f})')
    
    # Plot the curve outlines for clarity
    ax.plot(x, p_benign, color='darkblue', linewidth=2, alpha=0.8)
    ax.plot(x, p_pathogenic, color='darkred', linewidth=2, alpha=0.8)
    
    # Add data points overlay if available
    if evol_data is not None and len(evol_data) > 0:
        # Create histogram of actual data
        hist, bins = np.histogram(evol_data, bins=30, density=True)
        bin_centers = (bins[:-1] + bins[1:]) / 2
        ax.plot(bin_centers, hist * 0.1, 'k-', alpha=0.5, linewidth=1, label='Actual Data (scaled)')

    # Add vertical lines for means
    ax.axvline(x=params['mean_benign'], color='blue', linestyle=':', alpha=0.7, linewidth=2)
    ax.axvline(x=params['mean_pathogenic'], color='red', linestyle=':', alpha=0.7, linewidth=2)
    
    # Styling
    ax.set_title('GMM Decision Boundary: Benign vs Pathogenic Classification (with Means)', 
                fontsize=16, fontweight='bold', pad=20)
    ax.set_xlabel('Evolutionary Index', fontsize=14)
    ax.set_ylabel('Probability Density', fontsize=14)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=12, loc='upper right')
    ax.set_ylim(bottom=0)
    
    plt.tight_layout()
    
    # Save as PDF if filename ends with .pdf, otherwise PNG
    if str(output_file).endswith('.pdf'):
        plt.savefig(output_file, format='pdf', dpi=300, bbox_inches='tight')
    else:
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved GMM decision boundary plot with means to: {output_file}")

def create_classification_probability_plot(params, output_file):
    """Create a plot showing classification probabilities vs evolutionary index."""
    
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # Define the range for plotting
    x_min = min(params['mean_benign'] - 4*params['std_benign'], 
                params['mean_pathogenic'] - 4*params['std_pathogenic'])
    x_max = max(params['mean_benign'] + 4*params['std_benign'], 
                params['mean_pathogenic'] + 4*params['std_pathogenic'])
    
    x = np.linspace(x_min, x_max, 1000)
    
    # Calculate posterior probabilities
    p_pathogenic_given_x = params['weight_pathogenic'] * norm.pdf(x, params['mean_pathogenic'], params['std_pathogenic'])
    p_benign_given_x = params['weight_benign'] * norm.pdf(x, params['mean_benign'], params['std_benign'])
    
    # Normalize to get probabilities
    total_prob = p_pathogenic_given_x + p_benign_given_x
    p_pathogenic_posterior = p_pathogenic_given_x / total_prob
    p_benign_posterior = p_benign_given_x / total_prob
    
    # Plot probability curves
    ax.fill_between(x, 0, p_benign_posterior, alpha=0.6, color='dodgerblue', label='P(Benign | Evol Index)')
    ax.fill_between(x, p_benign_posterior, 1, alpha=0.6, color='crimson', label='P(Pathogenic | Evol Index)')
    
    # Add decision boundary
    boundaries = find_decision_boundary(params)
    for boundary in boundaries:
        ax.axvline(x=boundary, color='black', linestyle='--', linewidth=3, alpha=0.8)
        ax.annotate(f'50% Boundary\n{boundary:.2f}', 
                   xy=(boundary, 0.5), 
                   xytext=(boundary + 0.5, 0.3),
                   fontsize=11, fontweight='bold',
                   arrowprops=dict(arrowstyle='->', color='black', alpha=0.7))
    
    # Styling
    ax.set_title('Classification Probabilities vs Evolutionary Index', 
                fontsize=16, fontweight='bold', pad=20)
    ax.set_xlabel('Evolutionary Index', fontsize=14)
    ax.set_ylabel('Probability', fontsize=14)
    ax.set_ylim(0, 1)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=12)
    
    plt.tight_layout()
    
    # Save as PDF if filename ends with .pdf, otherwise PNG
    if str(output_file).endswith('.pdf'):
        plt.savefig(output_file, format='pdf', dpi=300, bbox_inches='tight')
    else:
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved classification probability plot to: {output_file}")

def create_gmm_with_data_histogram(params, output_file, eve_data=None, n_bins=50):
    """Create GMM decision boundary plot with colored histogram of actual data."""
    
    fig, ax = plt.subplots(figsize=(14, 8))
    
    # Define the range for plotting
    if eve_data is not None and len(eve_data) > 0:
        data_min, data_max = eve_data['EVE_scores'].min(), eve_data['EVE_scores'].max()
        x_min = min(params['mean_benign'] - 4*params['std_benign'], 
                    params['mean_pathogenic'] - 4*params['std_pathogenic'], 
                    data_min - 1)
        x_max = max(params['mean_benign'] + 4*params['std_benign'], 
                    params['mean_pathogenic'] + 4*params['std_pathogenic'], 
                    data_max + 1)
    else:
        x_min = min(params['mean_benign'] - 4*params['std_benign'], 
                    params['mean_pathogenic'] - 4*params['std_pathogenic'])
        x_max = max(params['mean_benign'] + 4*params['std_benign'], 
                    params['mean_pathogenic'] + 4*params['std_pathogenic'])
    
    # Create histogram of actual data colored by classification
    if eve_data is not None and len(eve_data) > 0:
        # Define bins
        bins = np.linspace(x_min, x_max, n_bins + 1)
        
        # Separate data by classification
        benign_data = eve_data[eve_data['EVE_classes_100_pct_retained'] == 'Benign']['EVE_scores']
        pathogenic_data = eve_data[eve_data['EVE_classes_100_pct_retained'] == 'Pathogenic']['EVE_scores']
        
        # Create stacked histogram
        ax.hist([benign_data, pathogenic_data], bins=bins, alpha=0.7, 
               color=['dodgerblue', 'crimson'], 
               label=[f'Benign (n={len(benign_data)})', f'Pathogenic (n={len(pathogenic_data)})'],
               edgecolor='black', linewidth=0.5)
        
        print(f"Histogram created with {len(benign_data)} benign and {len(pathogenic_data)} pathogenic sequences")
    
    # Overlay GMM curves
    x = np.linspace(x_min, x_max, 1000)
    
    # Calculate the probability densities and scale them to match histogram
    p_pathogenic = params['weight_pathogenic'] * norm.pdf(x, params['mean_pathogenic'], params['std_pathogenic'])
    p_benign = params['weight_benign'] * norm.pdf(x, params['mean_benign'], params['std_benign'])
    
    # Scale the curves to match the histogram scale
    if eve_data is not None and len(eve_data) > 0:
        bin_width = (x_max - x_min) / n_bins
        total_sequences = len(eve_data)
        scale_factor = total_sequences * bin_width
        
        p_pathogenic_scaled = p_pathogenic * scale_factor
        p_benign_scaled = p_benign * scale_factor
    else:
        p_pathogenic_scaled = p_pathogenic * 100  # Arbitrary scaling for demo
        p_benign_scaled = p_benign * 100
        scale_factor = 100
    
    # Plot GMM curves
    ax.plot(x, p_benign_scaled, color='darkblue', linewidth=3, alpha=0.9)
    ax.plot(x, p_pathogenic_scaled, color='darkred', linewidth=3, alpha=0.9)
    
    # Styling
    ax.set_title('Sequence Distribution with GMM Model Fit\n(Bars colored by EVE Classification)', 
                fontsize=16, fontweight='bold', pad=20)
    ax.set_xlabel('Evolutionary Index (EVE Score)', fontsize=14)
    ax.set_ylabel('Number of Sequences', fontsize=14)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=12, loc='upper right')
    
    plt.tight_layout()
    
    # Save as PDF if filename ends with .pdf, otherwise PNG
    if str(output_file).endswith('.pdf'):
        plt.savefig(output_file, format='pdf', dpi=300, bbox_inches='tight')
    else:
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved GMM with data histogram to: {output_file}")

# This is the plain histogram with GMM overlay (NO means lines)
def create_plain_histogram_with_gmm_overlay(params, scores_data, output_file, n_bins=50):
    """
    Create a plot with a histogram of all scores and overlay the GMM components.
    (No mean lines for GMM components in this version)

    Args:
        params (dict): Dictionary containing GMM parameters.
        scores_data (np.array): Array of evolutionary indices (EVE scores).
        output_file (Path or str): Path to save the output plot.
        n_bins (int): Number of bins for the histogram.
    """
    if scores_data is None or scores_data.size == 0:
        output_filename = Path(output_file).name 
        print(f"Skipping plain histogram with GMM overlay for {output_filename}: EVE scores data is missing or empty.")
        return

    with plt.style.context('seaborn-v0_8-white'):
        fig, ax = plt.subplots(figsize=(12, 8))

        data_min, data_max = scores_data.min(), scores_data.max()
        gmm_min_bound = min(params['mean_benign'] - 4 * params['std_benign'],
                            params['mean_pathogenic'] - 4 * params['std_pathogenic'])
        gmm_max_bound = max(params['mean_benign'] + 4 * params['std_benign'],
                            params['mean_pathogenic'] + 4 * params['std_pathogenic'])
        x_plot_min = min(data_min, gmm_min_bound) - 1 
        x_plot_max = max(data_max, gmm_max_bound) + 1

        ax.hist(scores_data, bins=n_bins, range=(x_plot_min, x_plot_max), alpha=0.6, 
                label='_nolegend_', color='grey', edgecolor='darkgrey', linewidth=0.5)
        
        bin_width = (x_plot_max - x_plot_min) / n_bins

        x_gmm = np.linspace(x_plot_min, x_plot_max, 1000)
        pdf_benign = params['weight_benign'] * norm.pdf(x_gmm, params['mean_benign'], params['std_benign'])
        pdf_pathogenic = params['weight_pathogenic'] * norm.pdf(x_gmm, params['mean_pathogenic'], params['std_pathogenic'])
        scale_factor = len(scores_data) * bin_width
        scaled_pdf_benign = pdf_benign * scale_factor
        scaled_pdf_pathogenic = pdf_pathogenic * scale_factor
        
        ax.fill_between(x_gmm, scaled_pdf_benign, color='dodgerblue', alpha=0.6, label='Benign')
        ax.fill_between(x_gmm, scaled_pdf_pathogenic, color='crimson', alpha=0.6, label='Pathogenic')
        ax.plot(x_gmm, scaled_pdf_benign, color='darkblue', linewidth=1.5, alpha=0.7)
        ax.plot(x_gmm, scaled_pdf_pathogenic, color='darkred', linewidth=1.5, alpha=0.7)

        ax.set_xlabel('Evolutionary Index', fontsize=28)
        ax.set_ylabel('Number of Sequences', fontsize=28)
        ax.legend(fontsize=28, loc='upper right')
        ax.set_ylim(bottom=0)
        ax.tick_params(axis='both', which='major', labelsize=24)

        # Remove top and right spines
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        
        plt.tight_layout()
        
        if str(output_file).endswith('.pdf'):
            plt.savefig(output_file, format='pdf', dpi=300, bbox_inches='tight')
        else:
            plt.savefig(output_file, dpi=300, bbox_inches='tight')
        plt.close(fig)
        output_filename = Path(output_file).name
        print(f"Saved plain histogram with GMM overlay to: {output_filename}")

# This is the plain histogram with GMM overlay AND means lines
def create_plain_histogram_with_gmm_means_overlay(params, scores_data, output_file, n_bins=50):
    """
    Create a plot with a histogram of all scores, overlay the GMM components,
    and add vertical lines for the GMM means.

    Args:
        params (dict): Dictionary containing GMM parameters.
        scores_data (np.array): Array of evolutionary indices (EVE scores).
        output_file (Path or str): Path to save the output plot.
        n_bins (int): Number of bins for the histogram.
    """
    if scores_data is None or scores_data.size == 0:
        output_filename = Path(output_file).name 
        print(f"Skipping plain histogram with GMM means overlay for {output_filename}: EVE scores data is missing or empty.")
        return

    with plt.style.context('seaborn-v0_8-white'):
        fig, ax = plt.subplots(figsize=(12, 8))

        data_min, data_max = scores_data.min(), scores_data.max()
        gmm_min_bound = min(params['mean_benign'] - 4 * params['std_benign'],
                            params['mean_pathogenic'] - 4 * params['std_pathogenic'])
        gmm_max_bound = max(params['mean_benign'] + 4 * params['std_benign'],
                            params['mean_pathogenic'] + 4 * params['std_pathogenic'])
        x_plot_min = min(data_min, gmm_min_bound) - 1 
        x_plot_max = max(data_max, gmm_max_bound) + 1

        ax.hist(scores_data, bins=n_bins, range=(x_plot_min, x_plot_max), alpha=0.6, 
                label='_nolegend_', color='grey', edgecolor='darkgrey', linewidth=0.5)
        
        bin_width = (x_plot_max - x_plot_min) / n_bins 

        x_gmm = np.linspace(x_plot_min, x_plot_max, 1000)
        pdf_benign = params['weight_benign'] * norm.pdf(x_gmm, params['mean_benign'], params['std_benign'])
        pdf_pathogenic = params['weight_pathogenic'] * norm.pdf(x_gmm, params['mean_pathogenic'], params['std_pathogenic'])
        scale_factor = len(scores_data) * bin_width
        scaled_pdf_benign = pdf_benign * scale_factor
        scaled_pdf_pathogenic = pdf_pathogenic * scale_factor
        
        ax.fill_between(x_gmm, scaled_pdf_benign, color='dodgerblue', alpha=0.6, label='Benign')
        ax.fill_between(x_gmm, scaled_pdf_pathogenic, color='crimson', alpha=0.6, label='Pathogenic')
        ax.plot(x_gmm, scaled_pdf_benign, color='darkblue', linewidth=1.5, alpha=0.7)
        ax.plot(x_gmm, scaled_pdf_pathogenic, color='darkred', linewidth=1.5, alpha=0.7)

        # Add vertical lines for GMM means
        ax.axvline(params['mean_benign'], color='blue', linestyle='--', linewidth=2, alpha=0.8, label=f'Benign Mean: {params["mean_benign"]:.2f}')
        ax.axvline(params['mean_pathogenic'], color='red', linestyle='--', linewidth=2, alpha=0.8, label=f'Pathogenic Mean: {params["mean_pathogenic"]:.2f}')

        ax.set_xlabel('Evolutionary Index', fontsize=16)
        ax.set_ylabel('Number of Sequences', fontsize=16)
        ax.legend(fontsize=16, loc='upper right')
        ax.set_ylim(bottom=0)
        ax.tick_params(axis='both', which='major', labelsize=16)

        # Remove top and right spines
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        
        plt.tight_layout()
        
        if str(output_file).endswith('.pdf'):
            plt.savefig(output_file, format='pdf', dpi=300, bbox_inches='tight')
        else:
            plt.savefig(output_file, dpi=300, bbox_inches='tight')
        plt.close(fig)
        output_filename = Path(output_file).name
        print(f"Saved plain histogram with GMM means overlay to: {output_filename}")

def create_classification_pie_chart(eve_data, output_file):
    """
    Create a pie chart showing the proportion of Benign vs Pathogenic classifications.

    Args:
        eve_data (pd.DataFrame): DataFrame containing 'EVE_classes_100_pct_retained'.
        output_file (Path or str): Path to save the output plot.
    """
    if eve_data is None or eve_data.empty or 'EVE_classes_100_pct_retained' not in eve_data.columns:
        output_filename = Path(output_file).name
        print(f"Skipping classification pie chart for {output_filename}: EVE data is missing, empty, or lacks classification column.")
        return

    plt.style.use('seaborn-v0_8-darkgrid') # Ensure style is consistent
    fig, ax = plt.subplots(figsize=(8, 8)) # Square figure size is often good for pie charts

    class_counts = eve_data['EVE_classes_100_pct_retained'].value_counts()
    
    # Ensure we have counts for both Benign and Pathogenic, even if one is zero for robustness
    labels = ['Benign', 'Pathogenic']
    sizes = [class_counts.get('Benign', 0), class_counts.get('Pathogenic', 0)]
    colors = ['#5A9BD5', '#B82E40'] # Dull_dodgerblue, Dull_crimson
    explode = (0, 0.05)  # Slightly explode the Pathogenic slice if desired, or (0,0) for no explosion

    # Filter out zero-sized slices to prevent issues with autopct if only one class exists
    valid_indices = [i for i, size in enumerate(sizes) if size > 0]
    labels = [labels[i] for i in valid_indices]
    sizes = [sizes[i] for i in valid_indices]
    colors = [colors[i] for i in valid_indices]
    explode = [explode[i] for i in valid_indices]

    if not sizes: # If no valid data after filtering
        output_filename = Path(output_file).name
        print(f"Skipping classification pie chart for {output_filename}: No data to plot after filtering.")
        plt.close(fig)
        return

    wedges, texts, autotexts = ax.pie(
        sizes,
        explode=explode,
        labels=labels,
        colors=colors,
        autopct='%1.1f%%',
        shadow=False,
        startangle=90,
        textprops=dict(color="black", fontsize=20, fontweight='bold')
    )

    # Improve autopct (percentage text) appearance
    for autotext in autotexts:
        autotext.set_color('white')
        autotext.set_fontsize(20)
        autotext.set_fontweight('bold')

    ax.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.
    
    # No need for plt.tight_layout() usually with single pie chart, but can be kept
    plt.tight_layout()

    if str(output_file).endswith('.pdf'):
        plt.savefig(output_file, format='pdf', dpi=300, bbox_inches='tight')
    else:
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close(fig)
    output_filename = Path(output_file).name
    print(f"Saved classification pie chart to: {output_filename}")

def create_scores_boxplot(eve_data, output_file):
    """
    Create a box plot of EVE scores for Benign vs Pathogenic classes with specific styling.

    Args:
        eve_data (pd.DataFrame): DataFrame containing scores and classifications.
        output_file (Path or str): Path to save the output plot.
    """
    if eve_data is None or eve_data.empty:
        output_filename = Path(output_file).name
        print(f"Skipping scores box plot for {output_filename}: EVE data is missing or empty.")
        return

    required_classification_col = 'EVE_classes_100_pct_retained'
    if required_classification_col not in eve_data.columns:
        output_filename = Path(output_file).name
        print(f"Skipping scores box plot for {output_filename}: Missing required column '{required_classification_col}'.")
        return

    # Determine which score column to use for the boxplot
    score_col_for_boxplot = 'EVE_scores' # Default to the GMM score column
    if 'EVE_scores_for_boxplot' in eve_data.columns:
        print(f"Using 'EVE_scores_for_boxplot' column for box plot.")
        score_col_for_boxplot = 'EVE_scores_for_boxplot'
    elif 'EVE_scores' in eve_data.columns: 
        print(f"'EVE_scores_for_boxplot' not found, falling back to 'EVE_scores' (GMM scores) for box plot.")
    else:
        output_filename = Path(output_file).name
        print(f"Skipping scores box plot for {output_filename}: No suitable score column found in EVE data ('{score_col_for_boxplot}').")
        return

    with plt.style.context('seaborn-v0_8-white'):
        fig, ax = plt.subplots(figsize=(8, 7)) 

        palette = {
            "Benign": "dodgerblue",
            "Pathogenic": "crimson"
        }

        sns.boxplot(
            x=required_classification_col,
            y=score_col_for_boxplot,
            data=eve_data,
            palette=palette,
            ax=ax,
            width=0.5,
            showfliers=False
        )

        # Overlay actual data points
        sns.stripplot(
            x=required_classification_col,
            y=score_col_for_boxplot,
            data=eve_data,
            ax=ax,
            size=3, # Small size for points
            color='black', # Color of the points
            alpha=0.5, # Transparency
            jitter=True # Add jitter to avoid overplotting
        )

        # Styling
        ax.set_xlabel('Predicted Class', fontsize=26, labelpad=15)
        ax.set_ylabel('EVE Score', fontsize=26, labelpad=15)
        ax.tick_params(axis='both', which='major', labelsize=24)
        
        # No grid lines: ax.grid(...) is not called

        # Remove top and right spines
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        
        plt.tight_layout()
        
        if str(output_file).endswith('.pdf'):
            plt.savefig(output_file, format='pdf', dpi=300, bbox_inches='tight')
        else:
            plt.savefig(output_file, dpi=300, bbox_inches='tight')
        plt.close(fig)
        output_filename = Path(output_file).name
        print(f"Saved scores box plot to: {output_filename}")

def main():
    parser = argparse.ArgumentParser(description='Visualize GMM decision boundary for benign vs pathogenic classification')
    parser.add_argument('--analysis_dir', type=str, default='.',
                        help='Directory containing ensemble analysis results')
    parser.add_argument('--output_dir', type=str, default='gmm_plots',
                        help='Output directory for plots')
    parser.add_argument('--simulate', action='store_true',
                        help='Use simulated GMM parameters for demonstration')
    parser.add_argument('--include_data', action='store_true',
                        help='Include actual data points overlay if available')
    parser.add_argument('--histogram_only', action='store_true',
                        help='Create only the histogram plot with classifications')
    parser.add_argument('--n-bins', type=int, default=75,
                        help='Number of bins for histograms (default: 75)')
    
    args = parser.parse_args()
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True)
    
    print("="*60)
    print("GMM DECISION BOUNDARY VISUALIZATION")
    print("="*60)
    
    # Load GMM parameters
    if args.simulate:
        params = simulate_gmm_parameters()
    else:
        gmm_file = find_gmm_stats_file(args.analysis_dir)
        if gmm_file:
            params = load_gmm_parameters(gmm_file)
        else:
            print("No GMM stats file found. Using simulated parameters.")
            params = simulate_gmm_parameters()
    
    if params is None:
        print("ERROR: Could not load GMM parameters")
        return
    
    # Load EVE scores with classifications for histogram
    eve_data = load_eve_scores_with_classifications(args.analysis_dir)
    
    # Load evol indices data for overlay (original functionality)
    evol_data = None
    if args.include_data:
        evol_data = load_evol_indices_data(args.analysis_dir)
    
    # Create visualizations
    print(f"\nCreating GMM decision boundary plots in: {output_dir}")
    
    # 1. New histogram plot with colored bars and GMM overlay
    if eve_data is not None:
        histogram_plot = output_dir / "gmm_data_histogram.png"
        create_gmm_with_data_histogram(params, histogram_plot, eve_data, n_bins=args.n_bins)
        
        histogram_plot_pdf = output_dir / "gmm_data_histogram.pdf"
        create_gmm_with_data_histogram(params, histogram_plot_pdf, eve_data, n_bins=args.n_bins)
        
        # Call the new plot function
        plain_hist_plot_png = output_dir / "plain_histogram_gmm_overlay.png"
        create_plain_histogram_with_gmm_overlay(params, eve_data['EVE_scores'].values, plain_hist_plot_png, n_bins=args.n_bins)
        
        plain_hist_plot_pdf = output_dir / "plain_histogram_gmm_overlay.pdf"
        create_plain_histogram_with_gmm_overlay(params, eve_data['EVE_scores'].values, plain_hist_plot_pdf, n_bins=args.n_bins)

        # Call the new plot function for histogram with GMM means
        plain_hist_means_plot_png = output_dir / "plain_histogram_gmm_means_overlay.png"
        create_plain_histogram_with_gmm_means_overlay(params, eve_data['EVE_scores'].values, plain_hist_means_plot_png, n_bins=args.n_bins)
        
        plain_hist_means_plot_pdf = output_dir / "plain_histogram_gmm_means_overlay.pdf"
        create_plain_histogram_with_gmm_means_overlay(params, eve_data['EVE_scores'].values, plain_hist_means_plot_pdf, n_bins=args.n_bins)

        # Call the new pie chart function (replacing box plot)
        pie_chart_png = output_dir / "classification_pie_chart.png"
        create_classification_pie_chart(eve_data, pie_chart_png)

        pie_chart_pdf = output_dir / "classification_pie_chart.pdf"
        create_classification_pie_chart(eve_data, pie_chart_pdf)

        # Add calls for the new/restyled box plot
        boxplot_png = output_dir / "scores_class_boxplot.png"
        create_scores_boxplot(eve_data, boxplot_png)

        boxplot_pdf = output_dir / "scores_class_boxplot.pdf"
        create_scores_boxplot(eve_data, boxplot_pdf)

    # 2. Original plots (unless histogram_only is specified)
    if not args.histogram_only:
        # Main decision boundary plot with filled curves
        main_plot = output_dir / "gmm_decision_boundary.png"
        create_gmm_decision_boundary_plot(params, main_plot, evol_data)
        
        # Also create PDF version
        main_plot_pdf = output_dir / "gmm_decision_boundary.pdf"
        create_gmm_decision_boundary_plot(params, main_plot_pdf, evol_data)

        # New: Decision boundary plot with mean lines
        means_plot = output_dir / "gmm_decision_boundary_with_means.png"
        create_gmm_decision_boundary_with_means_plot(params, means_plot, evol_data)

        means_plot_pdf = output_dir / "gmm_decision_boundary_with_means.pdf"
        create_gmm_decision_boundary_with_means_plot(params, means_plot_pdf, evol_data)
        
        # Classification probability plot
        prob_plot = output_dir / "classification_probabilities.png"
        create_classification_probability_plot(params, prob_plot)
        
        prob_plot_pdf = output_dir / "classification_probabilities.pdf"
        create_classification_probability_plot(params, prob_plot_pdf)
    
    # Print summary
    boundaries = find_decision_boundary(params)
    
    print(f"\n{'='*50}")
    print("GMM DECISION BOUNDARY SUMMARY")
    print(f"{'='*50}")
    
    print(f"\nGaussian Components:")
    print(f"  Benign: μ={params['mean_benign']:.3f}, σ={params['std_benign']:.3f}, π={params['weight_benign']:.3f}")
    print(f"  Pathogenic: μ={params['mean_pathogenic']:.3f}, σ={params['std_pathogenic']:.3f}, π={params['weight_pathogenic']:.3f}")
    
    if boundaries:
        print(f"\nDecision Boundary: {boundaries[0]:.3f}")
        print(f"  EVE Score < {boundaries[0]:.3f} → Classified as Benign")
        print(f"  EVE Score > {boundaries[0]:.3f} → Classified as Pathogenic")
    
    print(f"\nGenerated Files:")
    if eve_data is not None:
        print(f"  1. gmm_data_histogram.png/pdf - Sequence histogram colored by classification")
        print(f"  1b. plain_histogram_gmm_overlay.png/pdf - Plain sequence histogram with GMM component overlay")
        print(f"  1c. plain_histogram_gmm_means_overlay.png/pdf - Plain histogram with GMM overlay and means")
        print(f"  1d. classification_pie_chart.png/pdf - Pie chart of classifications")
        print(f"  1e. scores_class_boxplot.png/pdf - Box plot of scores by class")
    
    if not args.histogram_only:
        print(f"  2. gmm_decision_boundary.png/pdf - GMM curves with decision boundary")
        print(f"  2b. gmm_decision_boundary_with_means.png/pdf - GMM curves with mean lines")
        print(f"  3. classification_probabilities.png/pdf - Classification probability curves")

if __name__ == '__main__':
    main() 