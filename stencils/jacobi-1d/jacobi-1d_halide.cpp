#include "Halide.h"
#include <stdio.h>
#include <time.h>
#include <cmath>

using namespace Halide;

// Helper function to measure time
double get_time() {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec + ts.tv_nsec * 1e-9;
}

int main(int argc, char **argv) {
    // Define constants from the PolyBench implementation
    const int N = 2000; // This should match the value from jacobi-1d.h
    const int TSTEPS = 500; // This should match the value from jacobi-1d.h
    
    // Input arrays - initialize with same pattern as PolyBench
    Buffer<float> A_input(N);
    Buffer<float> B_input(N);
    
    // Initialize arrays with the same values as in the PolyBench implementation
    for (int i = 0; i < N; i++) {
        A_input(i) = ((float)i + 2) / N;
        B_input(i) = ((float)i + 3) / N;
    }
    
    // Create a copy for the naive implementation timing comparison
    Buffer<float> A_naive(N);
    Buffer<float> B_naive(N);
    
    // Manually copy the values
    for (int i = 0; i < N; i++) {
        A_naive(i) = A_input(i);
        B_naive(i) = B_input(i);
    }
    
    // Variables for Halide
    Var x("x");
    
    // Buffer to hold the current state of A and B
    Buffer<float> A_curr = A_input.copy();
    Buffer<float> B_curr = B_input.copy();
    
    // Time the Halide implementation
    double halide_start = get_time();
    
    // Implement the time loop outside of Halide
    for (int t = 0; t < TSTEPS; t++) {
        // Define the update for B
        Func B_update("B_update");
        B_update(x) = 0.33333f * (A_curr(x-1) + A_curr(x) + A_curr(x+1));
        
        // Schedule the B update
        B_update.vectorize(x, 8);
        B_update.parallel(x);
        
        // Create a temporary output buffer for B
        Buffer<float> B_temp(N);
        
        // Only update the valid range [1, N-2]
        // Keep boundary values unchanged
        B_temp(0) = B_curr(0);
        B_update.realize(B_temp.cropped(1, N-2));
        B_temp(N-1) = B_curr(N-1);
        
        // Copy the updated B values to B_curr
        B_curr = B_temp;
        
        // Define the update for A
        Func A_update("A_update");
        A_update(x) = 0.33333f * (B_curr(x-1) + B_curr(x) + B_curr(x+1));
        
        // Schedule the A update
        A_update.vectorize(x, 8);
        A_update.parallel(x);
        
        // Create a temporary output buffer for A
        Buffer<float> A_temp(N);
        
        // Only update the valid range [1, N-2]
        // Keep boundary values unchanged
        A_temp(0) = A_curr(0);
        A_update.realize(A_temp.cropped(1, N-2));
        A_temp(N-1) = A_curr(N-1);
        
        // Copy the updated A values to A_curr
        A_curr = A_temp;
    }
    
    double halide_end = get_time();
    double halide_time = halide_end - halide_start;
    
    // Run and time the naive implementation for comparison
    double naive_start = get_time();
    // Implement the same algorithm as in PolyBench
    for (int t = 0; t < TSTEPS; t++) {
        for (int i = 1; i < N - 1; i++) {
            B_naive(i) = 0.33333f * (A_naive(i-1) + A_naive(i) + A_naive(i+1));
        }
        for (int i = 1; i < N - 1; i++) {
            A_naive(i) = 0.33333f * (B_naive(i-1) + B_naive(i) + B_naive(i+1));
        }
    }
    double naive_end = get_time();
    double naive_time = naive_end - naive_start;
    
    // Verify correctness
    float max_diff = 0.0f;
    for (int i = 0; i < N; i++) {
        float diff = std::abs(A_curr(i) - A_naive(i));
        if (diff > max_diff) max_diff = diff;
    }
    
    // Print results
    printf("Halide implementation time: %0.6f seconds\n", halide_time);
    printf("Naive implementation time: %0.6f seconds\n", naive_time);
    printf("Speedup: %0.2fx\n", naive_time / halide_time);
    printf("Maximum difference: %e\n", max_diff);
    
    if (max_diff < 1e-4) {
        printf("Verification PASSED: Results match within threshold\n");
    } else {
        printf("Verification FAILED: Results differ beyond threshold\n");
    }
    
    return 0;
}