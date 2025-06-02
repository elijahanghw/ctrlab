import gymnasium as gym

class DroneRace(gym.Env):
    def __init__(self, drone, gates):
        self.drone = drone
        self.gates = gates
        
    def reset(self):
        pass
    
    def step(self):
        pass