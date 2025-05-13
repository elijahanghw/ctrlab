#pragma once

#include <cuda_runtime.h>
#include <math.h>

__device__ void vec_cross(float* a, float* b, float* result) {
    result[0] = a[1] * b[2] - a[2] * b[1];  
    result[1] = a[2] * b[0] - a[0] * b[2];  
    result[2] = a[0] * b[1] - a[1] * b[0];  
}

__device__ void mat_vec_mul(float* matrix, float* vector, float* result, int rows, int cols) {
    for (int i = 0; i < rows; i++) {
        result[i] = 0.0;
        for (int j = 0; j < cols; j++) {
            result[i] += matrix[i * cols + j] * vector[j];
        }
    }
}

__device__ void transpose(float* mat, float* result, int rows, int cols) {
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            result[j * rows + i] = mat[i * cols + j];
        }
    }
}