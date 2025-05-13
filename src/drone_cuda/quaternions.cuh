#pragma once

#include <cuda_runtime.h>
#include <math.h>

__device__ void quat_to_rot(float* q, float* R) {
    // Quaternion components
    float w = q[0];
    float x = q[1];
    float y = q[2];
    float z = q[3];

    // Compute the rotation matrix elements
    R[0] = 1.0f - 2.0f * (y * y + z * z); // R11
    R[1] = 2.0f * (x * y - z * w);     // R12
    R[2] = 2.0f * (x * z + y * w);     // R13

    R[3] = 2.0f * (x * y + z * w);     // R21
    R[4] = 1.0f - 2.0f * (x * x + z * z); // R22
    R[5] = 2.0f * (y * z - x * w);     // R23

    R[6] = 2.0f * (x * z - y * w);     // R31
    R[7] = 2.0f * (y * z + x * w);     // R32
    R[8] = 1.0f - 2.0f * (x * x + y * y); // R33
}

__device__ void quat_derivative(float* q, float* w, float* dq) {
    // Quaternion elements
    float q0 = q[0], q1 = q[1], q2 = q[2], q3 = q[3];
    // Angular velocity components
    float wx = w[0], wy = w[1], wz = w[2];

    // Compute quaternion derivative using 1/2 * Q(q) * omega
    dq[0] = 0.5f * ( -q1 * wx - q2 * wy - q3 * wz);
    dq[1] = 0.5f * (  q0 * wx + q2 * wz - q3 * wy);
    dq[2] = 0.5f * (  q0 * wy - q1 * wz + q3 * wx);
    dq[3] = 0.5f * (  q0 * wz + q1 * wy - q2 * wx);
}

__device__ void normalize_quaternion(float* q) {
    // Quaternion norm
    float norm = sqrtf(q[0]*q[0] + q[1]*q[1] + q[2]*q[2] + q[3]*q[3]);

    q[0] /= norm;
    q[1] /= norm;
    q[2] /= norm;
    q[3] /= norm;
}