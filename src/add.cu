#include <iostream>
#include <cmath>

// 1. kernel function
__global__
void add(int n, float *x, float *y)
{
    for (int i = 0; i < n; i++)
        y[i] = x[i] + y[i];
}

int main(void)
{
    int N = 1<<20; // 1M elements
    
    float *x, *y;
    // 2. Allocate Unified Memory – accessible from CPU or GPU
    cudaMallocManaged(&x, N*sizeof(float));
    cudaMallocManaged(&y, N*sizeof(float));
    
    // initialize x and y arrays on the host
    for (int i = 0; i < N; i++) {
        x[i] = 1.0f;
        y[i] = 2.0f;
    }
    
    // 2. Run kernel on 1M elements on the CPU
    add<<<1, 1>>>(N, x, y);

    // 2. Wait for GPU to finish before accessing on host
    cudaDeviceSynchronize();
    
    // Check for errors (all values should be 3.0f)
    float maxError = 0.0f;
    for (int i = 0; i < N; i++)
    maxError = fmax(maxError, fabs(y[i]-3.0f));
    std::cout << "Max error: " << maxError << std::endl;
    
    // 2. Free memory
    cudaFree(x);
    cudaFree(y);
    
    return 0;
}