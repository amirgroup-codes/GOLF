#!/usr/bin/env python3
"""
Ensemble Results Visualization Script

This script creates comprehensive visualizations to demonstrate how ensemble accuracy
improves with the number of models. It can work with results from the ensemble
analysis or simulate data for demonstration purposes.

Usage:
    python visualize_ensemble_results.py [--results_file results.csv] [--output_dir plots]
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import argparse
from pathlib import Path

# Set style for publication-quality plots
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

def load_or_simulate_data(results_file=None):
    """Load results from file or simulate realistic ensemble data."""
    
    if results_file and Path(results_file).exists():
        print(f"Loading results from: {results_file}")
        df = pd.read_csv(results_file)
        
        # Parse the analysis column to extract model numbers and ensemble sizes
        individual_data = []
        ensemble_data = []
        
        for _, row in df.iterrows():
            analysis = row['analysis']
            accuracy = row['accuracy']
            
            if analysis.startswith('model_'):
                model_num = int(analysis.split('_')[1])
                individual_data.append({
                    'model_number': model_num,
                    'accuracy': accuracy,
                    'type': 'Individual'
                })
            elif analysis.startswith('ensemble_'):
                ensemble_size = int(analysis.split('_')[1])
                ensemble_data.append({
                    'ensemble_size': ensemble_size,
                    'accuracy': accuracy,
                    'type': 'Ensemble'
                })
        
        individual_df = pd.DataFrame(individual_data)
        ensemble_df = pd.DataFrame(ensemble_data)
        
        return individual_df, ensemble_df
    
    else:
        print("Simulating realistic ensemble data for demonstration...")
        
        # Simulate individual model performance (with realistic variation)
        np.random.seed(42)  # For reproducible results
        
        # Individual models: varying performance with some being better than others
        individual_accuracies = [
            0.72 + np.random.normal(0, 0.02),  # Model 1
            0.75 + np.random.normal(0, 0.02),  # Model 2  
            0.70 + np.random.normal(0, 0.02),  # Model 3
            0.77 + np.random.normal(0, 0.02),  # Model 4
            0.73 + np.random.normal(0, 0.02),  # Model 5
        ]
        
        individual_df = pd.DataFrame({
            'model_number': range(1, 6),
            'accuracy': individual_accuracies,
            'type': 'Individual'
        })
        
        # Ensemble performance: generally improving with size, with diminishing returns
        base_accuracy = np.mean(individual_accuracies)
        ensemble_accuracies = []
        
        for size in range(2, 6):
            # Ensemble benefit with diminishing returns
            improvement = 0.08 * (1 - np.exp(-(size-1) * 0.5))
            noise = np.random.normal(0, 0.01)
            ensemble_acc = base_accuracy + improvement + noise
            ensemble_accuracies.append(min(ensemble_acc, 0.95))  # Cap at 95%
        
        ensemble_df = pd.DataFrame({
            'ensemble_size': range(2, 6),
            'accuracy': ensemble_accuracies,
            'type': 'Ensemble'
        })
        
        return individual_df, ensemble_df

def create_trend_plot(individual_df, ensemble_df, output_file):
    """Create the main trend plot showing ensemble improvement."""
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Plot 1: Individual Model Performance
    if not individual_df.empty:
        bars = ax1.bar(individual_df['model_number'], individual_df['accuracy'], 
                      alpha=0.7, color='skyblue', edgecolor='navy', linewidth=1.2)
        
        # Add value labels on bars
        for i, (idx, row) in enumerate(individual_df.iterrows()):
            ax1.annotate(f'{row["accuracy"]:.3f}', 
                        (row['model_number'], row['accuracy']), 
                        ha='center', va='bottom', fontweight='bold')
        
        ax1.set_title('Individual Model Performance', fontsize=14, fontweight='bold', pad=20)
        ax1.set_xlabel('Model Number', fontsize=12)
        ax1.set_ylabel('Accuracy', fontsize=12)
        ax1.set_ylim(0, 1)
        ax1.grid(True, alpha=0.3)
        ax1.set_xticks(individual_df['model_number'])
        
        # Add mean line
        mean_acc = individual_df['accuracy'].mean()
        ax1.axhline(y=mean_acc, color='red', linestyle='--', alpha=0.7, 
                   label=f'Average: {mean_acc:.3f}')
        ax1.legend()
    
    # Plot 2: Ensemble Performance Trend
    if not ensemble_df.empty:
        # Line plot with markers
        ax2.plot(ensemble_df['ensemble_size'], ensemble_df['accuracy'], 
                'ro-', linewidth=3, markersize=10, markerfacecolor='red', 
                markeredgecolor='darkred', markeredgewidth=2, label='Ensemble Accuracy')
        
        # Add value labels
        for _, row in ensemble_df.iterrows():
            ax2.annotate(f'{row["accuracy"]:.3f}', 
                        (row['ensemble_size'], row['accuracy']), 
                        textcoords="offset points", xytext=(0, 15), 
                        ha='center', fontweight='bold', fontsize=10)
        
        # Add individual model baseline for comparison
        if not individual_df.empty:
            best_individual = individual_df['accuracy'].max()
            ax2.axhline(y=best_individual, color='blue', linestyle='--', alpha=0.7, 
                       label=f'Best Individual: {best_individual:.3f}')
        
        ax2.set_title('Ensemble Accuracy Trend', fontsize=14, fontweight='bold', pad=20)
        ax2.set_xlabel('Number of Models in Ensemble', fontsize=12)
        ax2.set_ylabel('Accuracy', fontsize=12)
        ax2.set_ylim(0, 1)
        ax2.grid(True, alpha=0.3)
        ax2.set_xticks(ensemble_df['ensemble_size'])
        ax2.legend()
    
    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved trend plot to: {output_file}")

def create_improvement_plot(individual_df, ensemble_df, output_file):
    """Create a plot showing improvement over baseline."""
    
    if individual_df.empty or ensemble_df.empty:
        print("Skipping improvement plot - insufficient data")
        return
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Calculate improvement over best individual model
    best_individual = individual_df['accuracy'].max()
    mean_individual = individual_df['accuracy'].mean()
    
    improvements_over_best = (ensemble_df['accuracy'] - best_individual) * 100
    improvements_over_mean = (ensemble_df['accuracy'] - mean_individual) * 100
    
    # Create bar plot
    x = ensemble_df['ensemble_size']
    width = 0.35
    
    with plt.style.context('seaborn-v0_8-white'): # Apply white background style
        fig, ax = plt.subplots(figsize=(10, 6))

        bars1 = ax.bar(x - width/2, improvements_over_best, width, 
                       label='vs Best Individual', alpha=0.8, color='green', hatch='/') # Green with hatch
        bars2 = ax.bar(x + width/2, improvements_over_mean, width, 
                       label='vs Average Individual', alpha=0.8, color='#4682B4') # SteelBlue (duller blue), solid
        
        # Add value labels
        for bars_group in [bars1, bars2]:
            for bar in bars_group:
                height = bar.get_height()
                ax.annotate(f'{height:.1f}%',
                           xy=(bar.get_x() + bar.get_width() / 2, height),
                           xytext=(0, 3),  # 3 points vertical offset
                           textcoords="offset points",
                           ha='center', va='bottom', fontweight='bold', fontsize=11) # Increased fontsize for bar labels
        
        # ax.set_title('Ensemble Improvement Over Individual Models', fontsize=16, fontweight='bold', pad=20) # Removed title
        ax.set_xlabel('Number of Models in Ensemble', fontsize=14) # Increased xlabel fontsize
        ax.set_ylabel('Improvement (Percentage Points)', fontsize=14) # Increased ylabel fontsize
        ax.set_xticks(x)
        ax.tick_params(axis='both', which='major', labelsize=12) # Increased tick label size
        # ax.grid(True, alpha=0.3, axis='y') # Removed grid line
        ax.legend(fontsize=14) # Increased legend fontsize
        ax.axhline(y=0, color='black', linestyle='-', alpha=0.7) # Keep subtle baseline

        # Remove top and right spines
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        
        plt.tight_layout()
        
        # Save as PDF if the filename ends with .pdf, otherwise PNG
        if str(output_file).endswith('.pdf'):
            plt.savefig(output_file, format='pdf', dpi=300, bbox_inches='tight')
        else:
            plt.savefig(output_file, dpi=300, bbox_inches='tight')
        plt.close(fig)
    print(f"Saved improvement plot to: {output_file}")

def create_accuracy_plot(individual_df, ensemble_df, output_file):
    """Create a plot showing actual accuracy values with appropriate y-scale for high accuracy (90%+)."""
    
    if individual_df.empty or ensemble_df.empty:
        print("Skipping accuracy plot - insufficient data")
        return
    
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Find the overall accuracy range to set appropriate y-limits
    all_accuracies = []
    if not individual_df.empty:
        all_accuracies.extend(individual_df['accuracy'].tolist())
    if not ensemble_df.empty:
        all_accuracies.extend(ensemble_df['accuracy'].tolist())
    
    min_acc = min(all_accuracies)
    max_acc = max(all_accuracies)
    
    # Set y-limits with some padding - start from ~5% below minimum or 0.85, whichever is higher
    y_min = max(0.85, min_acc - 0.05)
    y_max = min(1.0, max_acc + 0.02)
    
    # Plot individual models as bars
    if not individual_df.empty:
        bars = ax.bar(individual_df['model_number'], individual_df['accuracy'], 
                     alpha=0.7, color='lightblue', edgecolor='navy', linewidth=1.5,
                     label='Individual Models', width=0.6)
        
        # Add value labels on bars
        for _, row in individual_df.iterrows():
            ax.annotate(f'{row["accuracy"]:.3f}\n({row["accuracy"]*100:.1f}%)', 
                       (row['model_number'], row['accuracy']), 
                       ha='center', va='bottom', fontweight='bold', fontsize=10)
    
    # Plot ensemble results as a line with markers
    if not ensemble_df.empty:
        ax.plot(ensemble_df['ensemble_size'], ensemble_df['accuracy'], 
               'ro-', linewidth=4, markersize=12, label='Ensemble', 
               markerfacecolor='red', markeredgecolor='darkred', markeredgewidth=2)
        
        # Add value labels for ensemble points
        for _, row in ensemble_df.iterrows():
            ax.annotate(f'{row["accuracy"]:.3f}\n({row["accuracy"]*100:.1f}%)', 
                       (row['ensemble_size'], row['accuracy']), 
                       textcoords="offset points", xytext=(0, 15), 
                       ha='center', fontweight='bold', fontsize=10, color='darkred')
    
    # Add horizontal lines for reference
    if not individual_df.empty:
        best_individual = individual_df['accuracy'].max()
        mean_individual = individual_df['accuracy'].mean()
        
        ax.axhline(y=best_individual, color='blue', linestyle='--', alpha=0.7, 
                  label=f'Best Individual: {best_individual:.3f}')
        ax.axhline(y=mean_individual, color='green', linestyle=':', alpha=0.7, 
                  label=f'Average Individual: {mean_individual:.3f}')
    
    # Styling
    ax.set_title('Model Accuracy Comparison: Individual vs Ensemble', 
                fontsize=16, fontweight='bold', pad=20)
    ax.set_xlabel('Model Configuration', fontsize=14)
    ax.set_ylabel('Accuracy', fontsize=14)
    ax.set_ylim(y_min, y_max)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=12)
    
    # Format y-axis to show percentages
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f'{y:.2f}\n({y*100:.1f}%)'))
    
    # Set x-ticks
    all_x = []
    if not individual_df.empty:
        all_x.extend(individual_df['model_number'].tolist())
    if not ensemble_df.empty:
        all_x.extend(ensemble_df['ensemble_size'].tolist())
    
    ax.set_xticks(sorted(set(all_x)))
    
    plt.tight_layout()
    
    # Save as PDF if the filename ends with .pdf, otherwise PNG
    if str(output_file).endswith('.pdf'):
        plt.savefig(output_file, format='pdf', dpi=300, bbox_inches='tight')
    else:
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved accuracy plot to: {output_file}")

def create_combined_comparison_plot(individual_df, ensemble_df, output_file):
    """Create a combined plot showing all data points."""
    
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Plot individual models
    if not individual_df.empty:
        ax.scatter(individual_df['model_number'], individual_df['accuracy'], 
                  s=120, alpha=0.7, color='blue', label='Individual Models', 
                  marker='s', edgecolors='darkblue', linewidth=2)
        
        # Add labels for individual models
        for _, row in individual_df.iterrows():
            ax.annotate(f'M{int(row["model_number"])}', 
                       (row['model_number'], row['accuracy']), 
                       textcoords="offset points", xytext=(0, -20), 
                       ha='center', fontweight='bold', color='darkblue')
    
    # Plot ensemble results
    if not ensemble_df.empty:
        ax.plot(ensemble_df['ensemble_size'], ensemble_df['accuracy'], 
               'ro-', linewidth=3, markersize=12, label='Ensemble', 
               markerfacecolor='red', markeredgecolor='darkred', markeredgewidth=2)
        
        # Add labels for ensemble points
        for _, row in ensemble_df.iterrows():
            ax.annotate(f'E{int(row["ensemble_size"])}', 
                       (row['ensemble_size'], row['accuracy']), 
                       textcoords="offset points", xytext=(0, 20), 
                       ha='center', fontweight='bold', color='darkred')
    
    # Styling
    ax.set_title('Individual vs Ensemble Model Performance', fontsize=16, fontweight='bold', pad=20)
    ax.set_xlabel('Model Configuration', fontsize=14)
    ax.set_ylabel('Accuracy', fontsize=14)
    ax.set_ylim(0, 1)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=12)
    
    # Custom x-axis to show both individual and ensemble points
    all_x = []
    all_labels = []
    
    if not individual_df.empty:
        all_x.extend(individual_df['model_number'].tolist())
        all_labels.extend([f'Model {i}' for i in individual_df['model_number']])
    
    if not ensemble_df.empty:
        all_x.extend(ensemble_df['ensemble_size'].tolist())
        all_labels.extend([f'Ensemble {i}' for i in ensemble_df['ensemble_size']])
    
    ax.set_xticks(sorted(set(all_x)))
    
    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved combined comparison plot to: {output_file}")

def create_statistical_summary_plot(individual_df, ensemble_df, output_file):
    """Create a statistical summary visualization."""
    
    if individual_df.empty or ensemble_df.empty:
        print("Skipping statistical summary - insufficient data")
        return
    
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
    
    # Plot 1: Distribution of individual model accuracies
    ax1.hist(individual_df['accuracy'], bins=5, alpha=0.7, color='skyblue', 
             edgecolor='black', linewidth=1.2)
    ax1.axvline(individual_df['accuracy'].mean(), color='red', linestyle='--', 
                label=f'Mean: {individual_df["accuracy"].mean():.3f}')
    ax1.set_title('Distribution of Individual Model Accuracies', fontweight='bold')
    ax1.set_xlabel('Accuracy')
    ax1.set_ylabel('Frequency')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Ensemble improvement trend with trend line
    if len(ensemble_df) > 1:
        x = ensemble_df['ensemble_size'].values
        y = ensemble_df['accuracy'].values
        
        # Fit a trend line
        z = np.polyfit(x, y, 1)
        p = np.poly1d(z)
        
        ax2.plot(x, y, 'ro-', linewidth=2, markersize=8, label='Actual')
        ax2.plot(x, p(x), 'b--', alpha=0.7, label=f'Trend (slope: {z[0]:.4f})')
        ax2.set_title('Ensemble Accuracy Trend with Fitted Line', fontweight='bold')
        ax2.set_xlabel('Ensemble Size')
        ax2.set_ylabel('Accuracy')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
    
    # Plot 3: Variance analysis
    individual_var = individual_df['accuracy'].var()
    if len(ensemble_df) > 1:
        ensemble_var = ensemble_df['accuracy'].var()
        
        categories = ['Individual Models', 'Ensemble Results']
        variances = [individual_var, ensemble_var]
        
        bars = ax3.bar(categories, variances, alpha=0.7, color=['skyblue', 'red'])
        ax3.set_title('Variance in Performance', fontweight='bold')
        ax3.set_ylabel('Variance')
        
        # Add value labels
        for bar, var in zip(bars, variances):
            ax3.annotate(f'{var:.5f}', (bar.get_x() + bar.get_width()/2, var), 
                        ha='center', va='bottom', fontweight='bold')
    
    # Plot 4: Summary statistics table
    ax4.axis('tight')
    ax4.axis('off')
    
    # Create summary statistics
    summary_data = []
    
    if not individual_df.empty:
        summary_data.extend([
            ['Individual Models', 'Count', f"{len(individual_df)}"],
            ['', 'Mean Accuracy', f"{individual_df['accuracy'].mean():.4f}"],
            ['', 'Std Dev', f"{individual_df['accuracy'].std():.4f}"],
            ['', 'Best', f"{individual_df['accuracy'].max():.4f}"],
            ['', 'Worst', f"{individual_df['accuracy'].min():.4f}"],
        ])
    
    if not ensemble_df.empty:
        summary_data.extend([
            ['Ensemble Results', 'Count', f"{len(ensemble_df)}"],
            ['', 'Mean Accuracy', f"{ensemble_df['accuracy'].mean():.4f}"],
            ['', 'Best', f"{ensemble_df['accuracy'].max():.4f}"],
            ['', 'Improvement', f"{(ensemble_df['accuracy'].max() - individual_df['accuracy'].max())*100:.2f}%"],
        ])
    
    table = ax4.table(cellText=summary_data, 
                     colLabels=['Category', 'Metric', 'Value'],
                     cellLoc='center', loc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.2, 1.5)
    
    ax4.set_title('Summary Statistics', fontweight='bold', pad=20)
    
    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved statistical summary to: {output_file}")

def main():
    parser = argparse.ArgumentParser(description='Visualize ensemble analysis results')
    parser.add_argument('--results_file', type=str, 
                        help='CSV file with ensemble analysis results')
    parser.add_argument('--output_dir', type=str, default='plots',
                        help='Output directory for plots')
    parser.add_argument('--simulate', action='store_true',
                        help='Use simulated data for demonstration')
    
    args = parser.parse_args()
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True)
    
    print("="*60)
    print("ENSEMBLE RESULTS VISUALIZATION")
    print("="*60)
    
    # Load or simulate data
    if args.simulate:
        individual_df, ensemble_df = load_or_simulate_data(None)
    else:
        # Try to find results file if not specified
        if not args.results_file:
            possible_files = [
                'ensemble_results/ensemble_analysis_results.csv',
                'results/ensemble_analysis_results.csv',
                'ensemble_analysis_results.csv'
            ]
            
            for f in possible_files:
                if Path(f).exists():
                    args.results_file = f
                    break
        
        individual_df, ensemble_df = load_or_simulate_data(args.results_file)
    
    # Create visualizations
    print(f"\nCreating visualizations in: {output_dir}")
    
    # 1. Main trend plot
    create_trend_plot(individual_df, ensemble_df, 
                     output_dir / "ensemble_trend_analysis.png")
    create_trend_plot(individual_df, ensemble_df, 
                     output_dir / "ensemble_trend_analysis.pdf")
    
    # 2. Improvement plot
    create_improvement_plot(individual_df, ensemble_df, 
                           output_dir / "ensemble_improvement.png")
    create_improvement_plot(individual_df, ensemble_df, 
                           output_dir / "ensemble_improvement.pdf")
    
    # 3. Accuracy plot
    create_accuracy_plot(individual_df, ensemble_df, 
                        output_dir / "ensemble_accuracy.png")
    
    # 4. Combined comparison
    create_combined_comparison_plot(individual_df, ensemble_df, 
                                   output_dir / "combined_comparison.png")
    
    # 5. Statistical summary
    create_statistical_summary_plot(individual_df, ensemble_df, 
                                   output_dir / "statistical_summary.png")
    
    # Print summary
    print(f"\n{'='*20} VISUALIZATION SUMMARY {'='*20}")
    
    if not individual_df.empty:
        print(f"Individual Models: {len(individual_df)} models")
        print(f"  Best accuracy: {individual_df['accuracy'].max():.3f}")
        print(f"  Mean accuracy: {individual_df['accuracy'].mean():.3f}")
        print(f"  Std deviation: {individual_df['accuracy'].std():.3f}")
    
    if not ensemble_df.empty:
        print(f"Ensemble Results: {len(ensemble_df)} configurations")
        print(f"  Best accuracy: {ensemble_df['accuracy'].max():.3f}")
        print(f"  Final accuracy: {ensemble_df['accuracy'].iloc[-1]:.3f}")
        
        if not individual_df.empty:
            improvement = (ensemble_df['accuracy'].max() - individual_df['accuracy'].max()) * 100
            print(f"  Improvement over best individual: {improvement:.2f} percentage points")
    
    print(f"\nAll plots saved to: {output_dir}")
    print("\nGenerated plots:")
    print("  1. ensemble_trend_analysis.png - Main trend comparison")
    print("  2. ensemble_improvement.png - Improvement analysis") 
    print("  3. ensemble_accuracy.png - Actual accuracy comparison")
    print("  4. combined_comparison.png - All data points together")
    print("  5. statistical_summary.png - Detailed statistics")

if __name__ == '__main__':
    main() 