import numpy as np

class SimpleQuad2DEnv:
    """
    Minimal quadrotor 2D environment.

    State (6):
      x = [px, pz, theta, vx, vz, omega]

    Action (2):
      u = [thrust, torque]

    Dynamics (very standard):
      px_dot = vx
      pz_dot = vz
      theta_dot = omega
      vx_dot = (T/m)*sin(theta)
      vz_dot = (T/m)*cos(theta) - g
      omega_dot = tau / Iyy

    This is enough for MPPI + autotuning pipeline.
    """

    def __init__(
        self,
        dt=0.02,
        max_steps=500,
        m=1.0,
        Iyy=0.02,
        g=9.81,
        thrust_min=0.0,
        thrust_max=25.0,
        torque_max=2.0,
        goal=np.array([2.0, 2.0, 0.0], dtype=np.float32),   # target: [px, pz, theta]
        pos_tol=0.15,
        theta_tol=0.2,
        seed=0,
    ):
        self.dt = float(dt)
        self.max_steps = int(max_steps)
        self.m = float(m)
        self.Iyy = float(Iyy)
        self.g = float(g)

        self.thrust_min = float(thrust_min)
        self.thrust_max = float(thrust_max)
        self.torque_max = float(torque_max)

        self.goal = np.array(goal, dtype=np.float32).reshape(3)
        self.pos_tol = float(pos_tol)
        self.theta_tol = float(theta_tol)

        self.rng = np.random.default_rng(seed)
        self.step_count = 0
        self.state = np.zeros(6, dtype=np.float32)

    @staticmethod
    def _wrap_angle(a):
        return (a + np.pi) % (2 * np.pi) - np.pi

    def reset(self):
        self.step_count = 0
        # Start near origin, slightly above ground
        px = 0.0
        pz = 1.0
        theta = 0.0

        vx, vz, omega = self.rng.normal(0.0, 0.05, size=3)
        self.state = np.array([px, pz, theta, vx, vz, omega], dtype=np.float32)
        return self.state.copy(), {}

    def step(self, action):
        self.step_count += 1
        u = np.asarray(action, dtype=np.float32).reshape(-1)
        if u.shape[0] != 2:
            raise ValueError(f"Expected action shape (2,), got {u.shape}")

        T, tau = float(u[0]), float(u[1])
        T = float(np.clip(T, self.thrust_min, self.thrust_max))
        tau = float(np.clip(tau, -self.torque_max, self.torque_max))

        px, pz, theta, vx, vz, omega = [float(v) for v in self.state]
        dt = self.dt

        # accelerations
        ax = (T / self.m) * np.sin(theta)
        az = (T / self.m) * np.cos(theta) - self.g
        alpha = tau / self.Iyy

        # integrate (semi-implicit Euler)
        vx = vx + ax * dt
        vz = vz + az * dt
        omega = omega + alpha * dt

        px = px + vx * dt
        pz = pz + vz * dt
        theta = self._wrap_angle(theta + omega * dt)

        self.state = np.array([px, pz, theta, vx, vz, omega], dtype=np.float32)

        # --- reward (negative cost) ---
        # goal is [px, pz, theta]
        e_pos = np.linalg.norm(self.state[0:2] - self.goal[0:2])
        e_th = abs(self._wrap_angle(theta - float(self.goal[2])))
        u2 = T*T + tau*tau

        # reward = -(pos error + angle error + control effort)
        reward = -(5.0 * e_pos + 1.0 * e_th + 0.001 * u2)

        done = False
        info = {"e_pos": float(e_pos), "e_theta": float(e_th)}

        # success condition
        if e_pos < self.pos_tol and e_th < self.theta_tol:
            done = True
            reward += 50.0
            info["success"] = True
        else:
            info["success"] = False

        # crash if below ground
        if pz < 0.0:
            done = True
            reward -= 50.0
            info["crash"] = True
        else:
            info["crash"] = False

        # time limit
        if self.step_count >= self.max_steps:
            done = True
            info["time_limit"] = True
        else:
            info["time_limit"] = False

        return self.state.copy(), float(reward), bool(done), info
