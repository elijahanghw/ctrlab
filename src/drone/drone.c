#include <string.h>
#include "drone.h"
#include "matrix.h"
#include "quaternions.h"

#define NUM_STATES 13

void equations_of_motion(float mass, float* I, float* I_inv, float* G1, float* states, float* U, float* derivatives) {
    float *Q = &states[3];
    float *V = &states[7];
    float *W = &states[10];

    // Calculate forces
    float F[6], rpm[4]; 
    for (int i = 0; i < 4; i++) {
        rpm[i] = U[i] * U[i];
    }
    mat_vec_mul(G1, rpm, F, 6, 4);

    // Compute rotation matrix
    float R[9], RT[9];
    quat_to_rot(Q, R);
    transpose(R, RT, 3, 3);

    // Calculate linear acceleration
    // Compute gravity
    float GRAVITY[3] = {0.0, 0.0, 9.81};
    float B_GRAVITY[3];
    mat_vec_mul(R, GRAVITY, B_GRAVITY, 3, 3);

    // Compute W x V
    float WxV[3];
    vec_cross(W, V, WxV);

    for (int i = 0; i < 3; i++) {
        derivatives[i + 7] = F[i] / mass - WxV[i] + B_GRAVITY[i];
    }

    // Calculate angular acceleration
    // Compute IW and WxIW
    float IW[3], WxIW[3];
    mat_vec_mul(I, W, IW, 3, 3);
    vec_cross(W, IW, WxIW);

    float alpha[3];
    for (int i = 0; i < 3; i++) {
        alpha[i] = F[i + 3] - WxIW[i];
    }

    mat_vec_mul(I_inv, alpha, &derivatives[10], 3, 3);

    // Calculate rate of change of position
    mat_vec_mul(RT, V, &derivatives[0], 3, 3);

    // Calculate quaternion derivatives
    quat_derivative(Q, W, &derivatives[3]);
}

void vec_equations_of_motion(int num_envs, float mass, float* I, float* I_inv, float* G1, float* states, float* U, float* derivatives) {
    for (int n = 0; n < num_envs; n++) {
        equations_of_motion(mass, I, I_inv, G1, &states[n*NUM_STATES], &U[n*4], &derivatives[n*NUM_STATES]);
    }
}


void integrate_rk4(float mass, float* I, float* I_inv, float* G1, float* states, float* U, float dt) {
    float temp_states[NUM_STATES];
    float c[4] = {0, 0.5, 0.5, 1};
    float k[4][NUM_STATES];

    for (int i = 0; i < NUM_STATES; i++) {
        temp_states[i] = states[i];
    }

    equations_of_motion(mass, I, I_inv, G1, temp_states, U, k[0]);

    for (int i = 1; i < 4; i++) { 
        for (int j = 0; j < NUM_STATES; j++) {
            temp_states[j] = states[j] + c[i] * k[i-1][j] * dt;
        }
        equations_of_motion(mass, I, I_inv, G1, temp_states, U, k[i]);
    }

    for (int i = 0; i < NUM_STATES; i++) {
        states[i] += (dt/6.0f) * (k[0][i] + 2.0f*k[1][i] + 2.0f*k[2][i] + k[3][i]);
    }

    normalize_quaternion(&states[3]);
}

void vec_integrate_rk4(int num_envs, float mass, float* I, float* I_inv, float* G1, float* states, float* U, float dt) {
    for (int n=0; n<num_envs; n++) {
        integrate_rk4(mass, I, I_inv, G1, &states[n*NUM_STATES], &U[n*4], dt);
    }
}

void integrate_euler(float mass, float* I, float* I_inv, float* G1, float* states, float* U, float dt) {
    float derivatives[NUM_STATES];

    equations_of_motion(mass, I, I_inv, G1, states, U, derivatives);
    for (int i = 0; i < NUM_STATES; i++) {
        states[i] += derivatives[i] * dt;
    }

    normalize_quaternion(&states[3]);
}

void vec_integrate_euler(int num_envs, float mass, float* I, float* I_inv, float* G1, float* states, float* U, float dt) {
    for (int n=0; n<num_envs; n++) {
        integrate_euler(mass, I, I_inv, G1, &states[n*NUM_STATES], &U[n*4], dt);
    }
}