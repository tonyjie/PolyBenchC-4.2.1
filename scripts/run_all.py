#!/usr/bin/env python3
# Compile and run all the benchmarks in the repository, collect the results, and save them to a file

import os
import subprocess
import re
import csv
import sys
from datetime import datetime
import matplotlib.pyplot as plt  # Add matplotlib for plotting

# Regex patterns to extract information from output
SPEEDUP_PATTERN = r'Speedup: (\d+\.\d+)'
VERIFICATION_PATTERN = r'Verification (PASSED|FAILED)'

# Define benchmarks and their directories
BENCHMARKS = {
    # datamining benchmarks
    "correlation": "datamining/correlation",
    "covariance": "datamining/covariance",
    # linear-algebra/blas benchmarks
    "gemm": "linear-algebra/blas/gemm",
    "gemver": "linear-algebra/blas/gemver",
    "gesummv": "linear-algebra/blas/gesummv",
    "symm": "linear-algebra/blas/symm",
    "syr2k": "linear-algebra/blas/syr2k",
    "syrk": "linear-algebra/blas/syrk",
    "trmm": "linear-algebra/blas/trmm",
    # linear-algebra/kernels benchmarks
    "2mm": "linear-algebra/kernels/2mm",
    "3mm": "linear-algebra/kernels/3mm",
    "atax": "linear-algebra/kernels/atax",
    "bicg": "linear-algebra/kernels/bicg",
    "doitgen": "linear-algebra/kernels/doitgen",
    "mvt": "linear-algebra/kernels/mvt",
    # linear-algebra/solvers benchmarks
    "cholesky": "linear-algebra/solvers/cholesky",
    "durbin": "linear-algebra/solvers/durbin",
    "gramschmidt": "linear-algebra/solvers/gramschmidt",
    "lu": "linear-algebra/solvers/lu",
    "ludcmp": "linear-algebra/solvers/ludcmp",
    "trisolv": "linear-algebra/solvers/trisolv",
    # medley benchmarks
    "deriche": "medley/deriche",
    "floyd-warshall": "medley/floyd-warshall",
    "nussinov": "medley/nussinov",
    # stencils benchmarks
    "adi": "stencils/adi",
    "fdtd-2d": "stencils/fdtd-2d",
    "heat-3d": "stencils/heat-3d",
    "jacobi-1d": "stencils/jacobi-1d",
    "jacobi-2d": "stencils/jacobi-2d",
    "seidel-2d": "stencils/seidel-2d",
}

# Function to run a command and capture output
def run_command(cmd, cwd=None, timeout=600):
    try:
        process = subprocess.Popen(
            cmd, 
            stdout=subprocess.PIPE, 
            stderr=subprocess.PIPE,
            cwd=cwd,
            text=True,
            shell=True
        )
        stdout, stderr = process.communicate(timeout=timeout)
        return process.returncode, stdout, stderr
    except subprocess.TimeoutExpired:
        process.kill()
        return -1, "", "Process timed out after {} seconds".format(timeout)
    except Exception as e:
        return -1, "", str(e)

def compile_benchmark(benchmark_dir):
    """Compile the benchmark using make"""
    result = {
        "compile_success": False,
        "compile_output": "",
        "compile_error": ""
    }
    
    returncode, stdout, stderr = run_command("make", cwd=benchmark_dir)
    result["compile_success"] = (returncode == 0)
    result["compile_output"] = stdout
    result["compile_error"] = stderr
    
    return result

def run_benchmark(benchmark_dir, binary_name):
    """Run the benchmark binary and collect results"""
    result = {
        "run_success": False,
        "run_output": "",
        "run_error": "",
        "verification": "UNKNOWN",
        "speedup": 0.0
    }
    
    # Path to the binary
    binary_path = f"./{binary_name}_mkl"
    
    returncode, stdout, stderr = run_command(binary_path, cwd=benchmark_dir)
    result["run_success"] = (returncode == 0)
    result["run_output"] = stdout
    result["run_error"] = stderr
    
    # Extract verification and speedup information
    if result["run_success"]:
        # Check for verification status
        verification_match = re.search(VERIFICATION_PATTERN, stdout)
        if verification_match:
            result["verification"] = verification_match.group(1)
        
        # Extract speedup
        speedup_match = re.search(SPEEDUP_PATTERN, stdout)
        if speedup_match:
            result["speedup"] = float(speedup_match.group(1))
    
    return result

def process_all_benchmarks():
    """Process all benchmarks and collect results"""
    results = []
    
    # Get repository root directory
    repo_root = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
    
    for benchmark_name, rel_path in BENCHMARKS.items():
        benchmark_dir = os.path.join(repo_root, rel_path)
        print(f"Processing benchmark: {benchmark_name} in {benchmark_dir}")
        
        result = {
            "benchmark": benchmark_name,
            "directory": rel_path,
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }
        
        # Compile the benchmark
        compile_result = compile_benchmark(benchmark_dir)
        result.update(compile_result)
        
        # Run the benchmark if compilation succeeded
        if compile_result["compile_success"]:
            run_result = run_benchmark(benchmark_dir, benchmark_name)
            result.update(run_result)
        
        results.append(result)
        
        print(f"  Compile success: {result['compile_success']}")
        print(f"  Run success: {result.get('run_success', False)}")
        print(f"  Verification: {result.get('verification', 'N/A')}")
        print(f"  Speedup: {result.get('speedup', 0.0)}")
        print("")
    
    return results

def save_results_to_csv(results, output_file):
    """Save results to a CSV file"""
    fieldnames = [
        "benchmark", "directory", "timestamp", 
        "compile_success", "run_success", 
        "verification", "speedup"
    ]
    
    with open(output_file, 'w', newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        
        for result in results:
            # Create a simplified row with just the fields we want
            row = {field: result.get(field, "") for field in fieldnames}
            writer.writerow(row)
    
    print(f"Results saved to {output_file}")

def save_detailed_results(results, output_dir):
    """Save detailed results including compile and run outputs to individual log files"""
    os.makedirs(output_dir, exist_ok=True)
    
    for result in results:
        benchmark = result["benchmark"]
        timestamp = result["timestamp"].replace(":", "-").replace(" ", "_")
        log_file = os.path.join(output_dir, f"{benchmark}_{timestamp}.log")
        
        with open(log_file, 'w') as f:
            f.write(f"Benchmark: {benchmark}\n")
            f.write(f"Directory: {result['directory']}\n")
            f.write(f"Timestamp: {result['timestamp']}\n")
            f.write(f"Compile Success: {result['compile_success']}\n")
            
            if not result['compile_success']:
                f.write("\nCompile Error:\n")
                f.write(result['compile_error'])
            
            if result.get('run_success') is not None:
                f.write(f"\nRun Success: {result['run_success']}\n")
                
                if not result['run_success']:
                    f.write("\nRun Error:\n")
                    f.write(result['run_error'])
                
                f.write("\nVerification: {}\n".format(result.get('verification', 'UNKNOWN')))
                f.write("Speedup: {}\n".format(result.get('speedup', 0.0)))
            
            f.write("\nFull Compile Output:\n")
            f.write(result['compile_output'])
            
            if 'run_output' in result:
                f.write("\nFull Run Output:\n")
                f.write(result['run_output'])
    
    print(f"Detailed logs saved to {output_dir}")

if __name__ == "__main__":
    # Create the results directory if it doesn't exist; results directory is in the same directory as the script
    results_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
    os.makedirs(results_dir, exist_ok=True)
    
    # Generate timestamp for output files
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Process all benchmarks
    results = process_all_benchmarks()
    
    # Save results to CSV file
    csv_file = os.path.join(results_dir, f"benchmark_results_{timestamp}.csv")
    save_results_to_csv(results, csv_file)
    
    # Save detailed results to logs directory
    logs_dir = os.path.join(results_dir, f"logs_{timestamp}")
    save_detailed_results(results, logs_dir)
        
    # Print summary
    print("\nBenchmark Summary:")
    print(f"{'Benchmark':<15} {'Compile':<10} {'Run':<10} {'Verification':<15} {'Speedup':<10}")
    print("-" * 60)
    
    for result in results:
        compile_status = "Success" if result["compile_success"] else "Fail"
        run_status = "Success" if result.get("run_success", False) else "Fail"
        verification = result.get("verification", "N/A")
        speedup = f"{result.get('speedup', 0.0):.2f}" if result.get("speedup", 0) > 0 else "N/A"
        
        print(f"{result['benchmark']:<15} {compile_status:<10} {run_status:<10} {verification:<15} {speedup:<10}")