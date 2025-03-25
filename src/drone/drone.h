#ifndef DRONE_H
#define DRONE_H

void equations_of_motion(float mass, float* I, float* I_inv, float* G1, float* states, float* U, float* derivatives) ;
void integrate_rk4(float mass, float* I, float* I_inv, float* G1, float* states, float* U, float dt);
void integrate_euler(float mass, float* I, float* I_inv, float* G1, float* states, float* U, float dt);

#endif