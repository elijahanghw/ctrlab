#ifndef DRONE_H
#define DRONE_H

void body_equations_of_motion(float mass, float* I, float* I_inv, float* G1, float tau, float* states, float* U, float* derivatives);
void inertial_equations_of_motion(float mass, float* I, float* I_inv, float* G1, float tau, float* states, float* U, float* derivatives);
void integrate_rk4(float mass, float* I, float* I_inv, float* G1, float tau, float* states, float* U, float dt);
void integrate_euler(float mass, float* I, float* I_inv, float* G1, float tau, float* states, float* U, float dt);

#endif