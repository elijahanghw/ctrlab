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

extern "C" void vec_integrate_rk4_deviceptr(int num_envs, float mass, float* d_I, float* d_I_inv, float* d_G1, float tau, float* d_states, float* d_inputs, float dt) {

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

}

extern "C" void vec_integrate_euler_deviceptr(int num_envs, float mass, float* d_I, float* d_I_inv, float* d_G1, float tau, float* d_states, float* d_inputs, float dt) {
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
}

// Struct for GPU Simulation State
extern "C" typedef struct {
    int num_envs;
    float* d_states;
    float* d_inputs;
    float* d_I;
    float* d_I_inv;
    float* d_G1;
} DroneSimGPU;


extern "C" void drone_sim_init(DroneSimGPU* sim, int num_envs) {
    sim->num_envs = num_envs;
    cudaMalloc(&sim->d_states, num_envs * NUM_STATES * sizeof(float));
    cudaMalloc(&sim->d_inputs, num_envs * 4 * sizeof(float));
    cudaMalloc(&sim->d_I, 9 * sizeof(float));
    cudaMalloc(&sim->d_I_inv, 9 * sizeof(float));
    cudaMalloc(&sim->d_G1, 6 * 4 * sizeof(float));
}

extern "C" void drone_sim_free(DroneSimGPU* sim) {
    cudaFree(sim->d_states);
    cudaFree(sim->d_inputs);
    cudaFree(sim->d_I);
    cudaFree(sim->d_I_inv);
    cudaFree(sim->d_G1);
}

extern "C" void drone_sim_step_rk4(DroneSimGPU* sim, float mass, float tau, float dt) {
    vec_integrate_rk4_deviceptr(sim->num_envs, mass, sim->d_I, sim->d_I_inv, sim->d_G1, tau, sim->d_states, sim->d_inputs, dt);
}

extern "C" void drone_sim_step_euler(DroneSimGPU* sim, float mass, float tau, float dt) {
    vec_integrate_euler_deviceptr(sim->num_envs, mass, sim->d_I, sim->d_I_inv, sim->d_G1, tau, sim->d_states, sim->d_inputs, dt);
}

extern "C" void drone_sim_copy_states_to_host(DroneSimGPU* sim, float* h_states) {
    cudaMemcpy(h_states, sim->d_states, sim->num_envs * NUM_STATES * sizeof(float), cudaMemcpyDeviceToHost);
}

extern "C" void drone_sim_copy_inputs_from_host(DroneSimGPU* sim, float* h_inputs) {
    cudaMemcpy(sim->d_inputs, h_inputs, sim->num_envs * 4 * sizeof(float), cudaMemcpyHostToDevice);
}

extern "C" void drone_sim_set_constants(DroneSimGPU* sim, float* h_I, float* h_I_inv, float* h_G1) {
    cudaMemcpy(sim->d_I, h_I, 9 * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(sim->d_I_inv, h_I_inv, 9 * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(sim->d_G1, h_G1, 6 * 4 * sizeof(float), cudaMemcpyHostToDevice);
}

extern "C" void drone_sim_set_states(DroneSimGPU* sim, float* h_states) {
    cudaMemcpy(sim->d_states, h_states, sim->num_envs * NUM_STATES * sizeof(float), cudaMemcpyHostToDevice);
}