#include "Halide.h"
#include <stdio.h>
#include <chrono>

using namespace Halide;

int main(int argc, char **argv) {
    // Problem size
    int n = 2000;              // Default array size
    int timesteps = 500;       // Default number of timesteps
    
    // Parse command line arguments if provided
    if (argc > 1) n = atoi(argv[1]);
    if (argc > 2) timesteps = atoi(argv[2]);
    
    // Initialize input data - same as the Polybench init_array
    Buffer<float> A(n);
    Buffer<float> B(n);
    
    for (int i = 0; i < n; i++) {
        A(i) = ((float)i + 2) / n;
        B(i) = ((float)i + 3) / n;
    }
    
    printf("Starting Jacobi-1D computation with n=%d, timesteps=%d\n", n, timesteps);
    
    // Halide variables
    Var x("x");
    
    // Timing
    auto start_time = std::chrono::high_resolution_clock::now();
    
    // Implement the Jacobi-1D kernel
    for (int t = 0; t < timesteps; t++) {
        // First pass: B[i] = 0.33333 * (A[i-1] + A[i] + A[i+1])
        Func b_update;
        // Use Halide's boundary conditions to handle the edges
        Expr clamped_x_minus_1 = clamp(x-1, 0, n-1);
        Expr clamped_x_plus_1 = clamp(x+1, 0, n-1);
        
        // This matches the Polybench implementation, computing only for interior points
        b_update(x) = select(
            x <= 0 || x >= n-1,  // Boundary condition
            B(x),                 // Keep original values at the edges
            0.33333f * (A(clamped_x_minus_1) + A(x) + A(clamped_x_plus_1)) // Update interior
        );
        
        // Realize and copy back to B buffer
        Buffer<float> b_result = b_update.realize({n});
        for (int i = 0; i < n; i++) {
            B(i) = b_result(i);
        }
        
        // Second pass: A[i] = 0.33333 * (B[i-1] + B[i] + B[i+1])
        Func a_update;
        a_update(x) = select(
            x <= 0 || x >= n-1,  // Boundary condition
            A(x),                 // Keep original values at the edges
            0.33333f * (B(clamped_x_minus_1) + B(x) + B(clamped_x_plus_1)) // Update interior
        );
        
        // Realize and copy back to A buffer
        Buffer<float> a_result = a_update.realize({n});
        for (int i = 0; i < n; i++) {
            A(i) = a_result(i);
        }
    }
    
    auto end_time = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed = end_time - start_time;
    
    printf("Jacobi-1D completed in: %g seconds\n", elapsed.count());
    
    // Print first and last few elements of the result
    printf("First 5 elements of A: ");
    for (int i = 0; i < std::min(5, n); i++) {
        printf("%f ", A(i));
    }
    printf("\nLast 5 elements of A: ");
    for (int i = std::max(0, n-5); i < n; i++) {
        printf("%f ", A(i));
    }
    printf("\n");
    
    return 0;
}