#!/usr/bin/env python3
import argparse
import os
import csv
import json
import matplotlib.pyplot as plt
import numpy as np

def create_speedup_barplot(csv_file, output_dir=None):
    """
    Create a bar plot of speedups with X marks for failures
    
    Args:
        csv_file: Path to the CSV file containing benchmark results
        output_dir: Directory where to save the plot (defaults to same dir as CSV file)
    """
    # Set default output directory if not provided
    if output_dir is None:
        output_dir = os.path.dirname(os.path.abspath(csv_file))
    
    # Load results from the CSV file
    results = []
    with open(csv_file, 'r', newline='') as f:
        reader = csv.DictReader(f)
        for row in reader:
            results.append(row)
    
    # Load MKL function usage data
    mkl_usage_file = os.path.join(output_dir, "mkl_function_usage.json")
    mkl_usage_data = {}
    if os.path.exists(mkl_usage_file):
        with open(mkl_usage_file, 'r') as f:
            mkl_usage_data = json.load(f)
    
    # Extract benchmark names and speedups
    benchmarks = []
    speedups = []
    failures = []
    mkl_calls_counts = []
    
    for result in results:
        benchmark_name = result["benchmark"]
        benchmarks.append(benchmark_name)
        
        # Get MKL calls count for this benchmark
        mkl_calls_count = 0
        if benchmark_name in mkl_usage_data:
            mkl_calls_count = mkl_usage_data[benchmark_name].get("num_math_calls", 0)
        mkl_calls_counts.append(mkl_calls_count)
        
        # Convert string values to appropriate types
        compile_success = result["compile_success"].lower() == 'true'
        run_success = result["run_success"].lower() == 'true'
        verification = result["verification"]
        
        # Check if benchmark failed at any stage
        failed = (not compile_success or 
                  not run_success or 
                  verification == "FAILED")
        
        # Set speedup value
        speedup_str = result["speedup"]
        if failed or not speedup_str or speedup_str == "N/A":
            speedups.append(np.nan)  # Set to NaN for failed benchmarks (won't display a bar)
        else:
            speedups.append(float(speedup_str))
        
        failures.append(failed)
    
    # Create the plot
    fig, ax = plt.subplots(figsize=(15, 8))
    
    # Create bars - NaN values won't create bars
    bars = ax.bar(benchmarks, speedups, color='skyblue')
    
    # y-axis in log scale
    ax.set_yscale('log')
    
    # Get y-axis limits to position the X marks correctly
    ymin, ymax = ax.get_ylim()
    
    # Add 'X' marks for failures at a visible position
    for i, failed in enumerate(failures):
        if failed:
            # For log scale, put the X at a fixed position above the x-axis
            ax.text(i, ymin * 1.3, 'X', 
                   ha='center',
                   va='center',
                   color='red',
                   fontweight='bold',
                   fontsize=16,
                   bbox=dict(facecolor='white', edgecolor='red', alpha=1.0, 
                            boxstyle="round,pad=0.3"))
    
    # Add speedup values and MKL call counts on top of each bar
    for i, (bar, speedup, mkl_count) in enumerate(zip(bars, speedups, mkl_calls_counts)):
        if not np.isnan(speedup):  # Only add text for valid speedups
            # Position speedup text just above the bar
            height = bar.get_height()
            ax.text(
                bar.get_x() + bar.get_width() / 2.,
                height * 1.05,  # Position slightly above the bar
                f'{speedup:.2f}',  # Format with 2 decimal places
                ha='center',
                va='bottom',
                fontsize=10,
                fontweight='bold',
                bbox=dict(facecolor='white', alpha=0.7, boxstyle="round,pad=0.1")
            )
            
            # Add MKL call count in red above the speedup text
            ax.text(
                bar.get_x() + bar.get_width() / 2.,
                height * 1.25,  # Position above the speedup text
                f'{mkl_count}',
                ha='center',
                va='bottom',
                fontsize=9,
                color='red',
                fontweight='bold',
                bbox=dict(facecolor='white', alpha=0.7, boxstyle="round,pad=0.1")
            )
    
    # Another horizontal line at y=1
    ax.axhline(y=1, color='black', linestyle='--')
    
    # Customize the plot
    ax.set_xlabel('Benchmarks')
    ax.set_ylabel('Speedup')
    ax.set_title('Kernel with MKL Library Calls vs. Naive Implementation\n(X marks indicate failures, red numbers show number of MKL math calls)')
    ax.set_xticklabels(benchmarks, rotation=45, ha='right')
    
    # Make sure the X labels fit properly
    plt.tight_layout()
    
    # Add a bottom margin to ensure the X markers are visible
    plt.subplots_adjust(bottom=0.2)
    
    # Save the plot
    plot_file = os.path.join(output_dir, "speedup_barplot.pdf")
    plt.savefig(plot_file)
    print(f"Speedup bar plot saved to {plot_file}")
    
    # Show the plot
    plt.close()

if __name__ == "__main__":
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Generate speedup barplot from benchmark results')
    parser.add_argument('csv_file', help='Path to the CSV file containing benchmark results')
    parser.add_argument('--output-dir', help='Directory where to save the plot')
    
    args = parser.parse_args()
    
    # Create the barplot
    create_speedup_barplot(args.csv_file, args.output_dir)
