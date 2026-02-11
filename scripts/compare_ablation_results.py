#!/usr/bin/env python3
"""
Compare ablation study results and generate performance table.
"""

import json
import argparse
from pathlib import Path
import pandas as pd
from typing import Dict, List


def load_results(results_path: Path) -> Dict:
    """Load results JSON file."""
    with open(results_path, 'r') as f:
        return json.load(f)


def extract_metrics(results: Dict) -> Dict:
    """Extract key metrics from results."""
    metrics = {}

    # Overall metrics from summary file
    if 'overall' in results:
        overall = results['overall']
        metrics['AUC'] = overall.get('avg_auc', 0.0)
        metrics['Precision@20'] = overall.get('avg_precision_20', 0.0)
        metrics['NormPrecision'] = overall.get('avg_normalized_precision', 0.0)
        metrics['AvgIoU'] = overall.get('avg_iou', 0.0)
        metrics['FPS'] = overall.get('avg_fps', 0.0)
        metrics['NumVideos'] = overall.get('num_videos', 0)

    return metrics


def main():
    parser = argparse.ArgumentParser(description='Compare ablation study results')
    parser.add_argument('--results-dir', type=str, required=True,
                       help='Directory containing results_A*.json files')
    parser.add_argument('--output', type=str, default='ablation_comparison.csv',
                       help='Output CSV file path')
    args = parser.parse_args()

    results_dir = Path(args.results_dir)

    # Find all ablation result directories
    result_dirs = sorted(results_dir.glob('results_A*'))

    # Filter only directories
    result_dirs = [d for d in result_dirs if d.is_dir()]

    if not result_dirs:
        print(f"No ablation results found in {results_dir}")
        return

    print(f"Found {len(result_dirs)} ablation experiments")
    print("")

    # Collect data
    data = []
    experiment_names = {
        'A1': 'Baseline (IoU only)',
        'A2': '+VPE',
        'A3': '+Kalman',
        'A4': '+VPE+Kalman',
        'A5': '+DIoU',
        'A6': '+Adaptive',
        'A7': '+DualMemory',
        'A8': 'Full (optimized)'
    }

    for result_dir in result_dirs:
        # Extract experiment ID from directory name
        exp_id = result_dir.stem.replace('results_', '').replace('.json', '')

        # Find summary JSON file inside directory (prefer summary over results)
        summary_files = list(result_dir.glob('summary_*.json'))

        if not summary_files:
            print(f"No summary file found in {result_dir}")
            continue

        result_file = summary_files[0]  # Take first matching summary file

        try:
            results = load_results(result_file)
            metrics = extract_metrics(results)

            row = {
                'Experiment': exp_id,
                'Description': experiment_names.get(exp_id, exp_id),
                **metrics
            }
            data.append(row)

        except Exception as e:
            print(f"Error processing {result_file}: {e}")

    # Create DataFrame
    df = pd.DataFrame(data)

    # Sort by experiment ID
    df = df.sort_values('Experiment')

    # Calculate improvements over baseline
    if len(df) > 0 and 'AUC' in df.columns:
        baseline_auc = df.iloc[0]['AUC']
        baseline_precision = df.iloc[0]['Precision@20']

        df['AUC Δ'] = df['AUC'] - baseline_auc
        df['Prec@20 Δ'] = df['Precision@20'] - baseline_precision

    # Display table
    print("Ablation Study Results:")
    print("=" * 100)
    print(df.to_string(index=False, float_format='%.4f'))
    print("")

    # Save to CSV
    output_path = Path(args.output)
    df.to_csv(output_path, index=False, float_format='%.6f')
    print(f"Results saved to: {output_path}")

    # Print insights
    print("")
    print("Key Insights:")
    print("-" * 100)

    if len(df) > 1 and 'AUC' in df.columns:
        best_exp = df.loc[df['AUC'].idxmax()]
        print(f"Best AUC: {best_exp['Experiment']} ({best_exp['Description']}) = {best_exp['AUC']:.4f}")

        best_prec = df.loc[df['Precision@20'].idxmax()]
        print(f"Best Precision@20: {best_prec['Experiment']} ({best_prec['Description']}) = {best_prec['Precision@20']:.4f}")

        fastest = df.loc[df['FPS'].idxmax()]
        print(f"Fastest: {fastest['Experiment']} ({fastest['Description']}) = {fastest['FPS']:.1f} FPS")

        if 'AUC Δ' in df.columns:
            improvements = df[df['AUC Δ'] > 0].sort_values('AUC Δ', ascending=False)
            if len(improvements) > 0:
                print("")
                print("Top improvements over baseline:")
                for _, row in improvements.head(3).iterrows():
                    print(f"  {row['Experiment']}: +{row['AUC Δ']:.4f} AUC, +{row['Prec@20 Δ']:.4f} precision@20")


if __name__ == '__main__':
    main()
