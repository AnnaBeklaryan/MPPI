import numpy as np
from mppi_torch import diffdrive_dynamics

class DiffDriveEnv:
    def __init__(self, dt=0.02, max_steps=2000, x0=None):
        self.dt = float(dt)
        self.max_steps = int(max_steps)
        self.k = 0
        self.x0 = np.array([0.0, 1.0, 0.0], dtype=np.float32) if x0 is None else np.asarray(x0, dtype=np.float32)
        self.x = self.x0.copy()

    def reset(self):
        self.k = 0
        self.x = self.x0.copy()
        return self.x.copy()

    def step(self, u):
        self.k += 1
        self.x = diffdrive_dynamics(self.x, u, self.dt).astype(np.float32)
        done = (self.k >= self.max_steps)
        reward = 0.0
        info = {}
        return self.x.copy(), reward, done, info