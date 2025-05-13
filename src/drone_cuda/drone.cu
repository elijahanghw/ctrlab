#include <cuda_runtime.h>
#include <math.h>
#include <stdio.h>

#include "matrix.cuh"
#include "quaternions.cuh"
#include "drone.cuh"


// Kernel
__global__ void rk4_kernel(int num_envs, float mass, float* I, float* I_inv, float* G1, float tau, float* states, float* U, float dt) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_envs) return;

    float* state = &states[idx * NUM_STATES];
    float* u = &U[idx * 4];

    integrate_rk4(mass, I, I_inv, G1, tau, state, u, dt);
}

__global__ void euler_kernel(int num_envs, float mass, float* I, float* I_inv, float* G1, float tau, float* states, float* U, float dt) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_envs) return;
    
    float* state = &states[idx * NUM_STATES];
    float* u = &U[idx * 4];

    integrate_euler(mass, I, I_inv, G1, tau, state, u, dt);
}

extern "C" void vec_integrate_rk4(int num_envs, float mass, float* h_I, float* h_I_inv, float* h_G1, float tau, float* h_states, float* h_inputs, float dt) {
    float *d_states, *d_inputs, *d_I, *d_I_inv, *d_G1;
    size_t state_size = num_envs * NUM_STATES * sizeof(float);
    size_t input_size = num_envs * 4 * sizeof(float);
    size_t mat3x3 = 9 * sizeof(float);
    size_t g1_size = 6 * 4 * sizeof(float); // shared G1

    // Allocate and copy to device
    cudaMalloc(&d_states, state_size);
    cudaMemcpy(d_states, h_states, state_size, cudaMemcpyHostToDevice);

    cudaMalloc(&d_inputs, input_size);
    cudaMemcpy(d_inputs, h_inputs, input_size, cudaMemcpyHostToDevice);

    cudaMalloc(&d_I, mat3x3);
    cudaMemcpy(d_I, h_I, mat3x3, cudaMemcpyHostToDevice);

    cudaMalloc(&d_I_inv, mat3x3);
    cudaMemcpy(d_I_inv, h_I_inv, mat3x3, cudaMemcpyHostToDevice);

    cudaMalloc(&d_G1, g1_size);
    cudaMemcpy(d_G1, h_G1, g1_size, cudaMemcpyHostToDevice);

    // Launch kernel
    int threads = 256;
    int blocks = (num_envs + threads - 1) / threads;
    rk4_kernel<<<blocks, threads>>>(
        num_envs, mass, d_I, d_I_inv, d_G1, tau, d_states, d_inputs, dt
    );

    cudaError_t error = cudaGetLastError();
    if (error != cudaSuccess) {
        printf("CUDA error: %s\n", cudaGetErrorString(error));
    }

    // Copy result back to host
    cudaMemcpy(h_states, d_states, state_size, cudaMemcpyDeviceToHost);

    // Free device memory
    cudaFree(d_states);
    cudaFree(d_inputs);
    cudaFree(d_I);
    cudaFree(d_I_inv);
    cudaFree(d_G1);
}

extern "C" void vec_integrate_euler(int num_envs, float mass, float* h_I, float* h_I_inv, float* h_G1, float tau, float* h_states, float* h_inputs, float dt) {
    float *d_states, *d_inputs, *d_I, *d_I_inv, *d_G1;
    size_t state_size = num_envs * NUM_STATES * sizeof(float);
    size_t input_size = num_envs * 4 * sizeof(float);
    size_t mat3x3 = 9 * sizeof(float);
    size_t g1_size = 6 * 4 * sizeof(float); // shared G1

    // Allocate and copy to device
    cudaMalloc(&d_states, state_size);
    cudaMemcpy(d_states, h_states, state_size, cudaMemcpyHostToDevice);

    cudaMalloc(&d_inputs, input_size);
    cudaMemcpy(d_inputs, h_inputs, input_size, cudaMemcpyHostToDevice);

    cudaMalloc(&d_I, mat3x3);
    cudaMemcpy(d_I, h_I, mat3x3, cudaMemcpyHostToDevice);

    cudaMalloc(&d_I_inv, mat3x3);
    cudaMemcpy(d_I_inv, h_I_inv, mat3x3, cudaMemcpyHostToDevice);

    cudaMalloc(&d_G1, g1_size);
    cudaMemcpy(d_G1, h_G1, g1_size, cudaMemcpyHostToDevice);

    // Launch kernel
    int threads = 256;
    int blocks = (num_envs + threads - 1) / threads;
    euler_kernel<<<blocks, threads>>>(
        num_envs, mass, d_I, d_I_inv, d_G1, tau, d_states, d_inputs, dt
    );

    cudaError_t error = cudaGetLastError();
    if (error != cudaSuccess) {
        printf("CUDA error: %s\n", cudaGetErrorString(error));
    }

    // Copy result back to host
    cudaMemcpy(h_states, d_states, state_size, cudaMemcpyDeviceToHost);

    // Free device memory
    cudaFree(d_states);
    cudaFree(d_inputs);
    cudaFree(d_I);
    cudaFree(d_I_inv);
    cudaFree(d_G1);
}
