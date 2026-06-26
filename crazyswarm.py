import numpy as np
import torch

class CrazyswarmEnv:
    def __init__(
        self,
        state_dim=9,
        action_dim=4,
        mass=0.028,
        Ixx=2.3951e-5,
        Iyy=2.3951e-5,
        Izz=3.2347e-5,
        cm=2.4e-6,
        attitude_time_constant=0.08,
        arm_length=0.044,
        gravity=9.80665,
        dt=0.02,
        max_episode_steps=3000,
        max_roll_pitch_rad=np.deg2rad(70.0),
        max_yaw_rad=np.pi,
        max_cmd_roll_pitch_rad=np.deg2rad(50.0),
        max_cmd_yaw_rad=np.deg2rad(50.0),
        min_thrust=0.0,
        max_thrust=0.4776,
        trajectory_type="helix",
        init_state = None,
        trajectory_bound_points=1000,
        trajectory_bound_margin=5,
        ground_effect=False,
        ground_effect_rotor_radius=0.0225,
        ground_effect_min_height=0.025,
        ground_effect_max_multiplier=1.25,
        hover_begin = 4.0,
        angle_freedom = False
    ):
        self.NAME = 'quadrotor_crazyfli'
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.mass = mass
        self.Ixx = Ixx
        self.Iyy = Iyy
        self.Izz = Izz
        self.cm = cm
        self.attitude_time_constant = attitude_time_constant
        self.arm_length = arm_length
        self.gravity = gravity
        self.tau = attitude_time_constant
        self.dt = dt
        self.max_episode_steps = max_episode_steps
        self.max_roll_pitch_rad = max_roll_pitch_rad
        self.max_yaw_rad = max_yaw_rad
        self.max_cmd_roll_pitch_rad = max_cmd_roll_pitch_rad
        self.max_cmd_yaw_rad = max_cmd_yaw_rad
        self.min_thrust = min_thrust
        self.max_thrust = max_thrust
        self.trajectory_type = trajectory_type
        self.trajectory_bound_points = trajectory_bound_points
        self.trajectory_bound_margin = trajectory_bound_margin
        self.ground_effect = ground_effect
        self.ground_effect_rotor_radius = ground_effect_rotor_radius
        self.ground_effect_min_height = ground_effect_min_height
        self.ground_effect_max_multiplier = ground_effect_max_multiplier
        self.hover_begin = hover_begin
        self.u_min = np.array(
            [
                -self.max_cmd_roll_pitch_rad,
                -self.max_cmd_roll_pitch_rad,
                -self.max_cmd_yaw_rad,
                self.min_thrust,
            ],
            dtype=np.float32,
        )
        self.u_max = np.array(
            [
                self.max_cmd_roll_pitch_rad,
                self.max_cmd_roll_pitch_rad,
                self.max_cmd_yaw_rad,
                self.max_thrust,
            ],
            dtype=np.float32,
        )
        
        self.steps = 0
        if init_state is None:
            self.state = np.zeros(self.state_dim, dtype=np.float32)
            self.trajectory_start_position = np.zeros(3, dtype=np.float32)
        else:
            init_state = np.asarray(init_state, dtype=np.float32)
            if init_state.shape != (self.state_dim,):
                raise ValueError(
                    f"Expected init_state with shape ({self.state_dim},), got {init_state.shape}."
                )
            self.state = init_state.copy()
            self.trajectory_start_position = self.state[:3].copy()
        self.obs = self.state.copy()
        self._update_trajectory_bounds()
       

    @staticmethod
    def _wrap_to_pi(angle):
        return (angle + np.pi) % (2.0 * np.pi) - np.pi

    def clip_action(self, action):
        action = np.asarray(action, dtype=np.float32)
        return np.clip(action, self.u_min, self.u_max)

    def clip_action_torch(self, action):
        u_min = torch.as_tensor(self.u_min, dtype=action.dtype, device=action.device)
        u_max = torch.as_tensor(self.u_max, dtype=action.dtype, device=action.device)
        return torch.clamp(action, min=u_min, max=u_max)

    def _ground_effect_multiplier(self, height):
        if not self.ground_effect:
            return 1.0
        h_eff = max(float(height), self.ground_effect_min_height)
        ratio = self.ground_effect_rotor_radius / (4.0 * h_eff)
        multiplier = 1.0 / max(1.0 - ratio * ratio, 1e-6)
        return min(multiplier, self.ground_effect_max_multiplier)

    def _ground_effect_multiplier_torch(self, height):
        if not self.ground_effect:
            return torch.ones_like(height)
        min_height = torch.as_tensor(
            self.ground_effect_min_height, dtype=height.dtype, device=height.device
        )
        rotor_radius = torch.as_tensor(
            self.ground_effect_rotor_radius, dtype=height.dtype, device=height.device
        )
        max_multiplier = torch.as_tensor(
            self.ground_effect_max_multiplier, dtype=height.dtype, device=height.device
        )
        one = torch.ones_like(height)
        eps = torch.as_tensor(1e-6, dtype=height.dtype, device=height.device)
        h_eff = torch.clamp(height, min=min_height)
        ratio = rotor_radius / (4.0 * h_eff)
        multiplier = 1.0 / torch.clamp(1.0 - ratio * ratio, min=eps)
        return torch.clamp(multiplier, min=one, max=max_multiplier)

    @staticmethod
    def _square_progress(t, side_length, speed):
        segment_time = side_length / speed
        cycle_time = 4.0 * segment_time
        t_cycle = t % cycle_time

        if t_cycle < segment_time:
            s = t_cycle / segment_time
            pos_uv = np.array([side_length * s, 0.0], dtype=np.float32)
            vel_uv = np.array([speed, 0.0], dtype=np.float32)
        elif t_cycle < 2.0 * segment_time:
            s = (t_cycle - segment_time) / segment_time
            pos_uv = np.array([side_length, side_length * s], dtype=np.float32)
            vel_uv = np.array([0.0, speed], dtype=np.float32)
        elif t_cycle < 3.0 * segment_time:
            s = (t_cycle - 2.0 * segment_time) / segment_time
            pos_uv = np.array([side_length * (1.0 - s), side_length], dtype=np.float32)
            vel_uv = np.array([-speed, 0.0], dtype=np.float32)
        else:
            s = (t_cycle - 3.0 * segment_time) / segment_time
            pos_uv = np.array([0.0, side_length * (1.0 - s)], dtype=np.float32)
            vel_uv = np.array([0.0, -speed], dtype=np.float32)

        return pos_uv, vel_uv

    def _update_trajectory_bounds(self):
        state1 = self.state
        ref = self.get_trejectory(N=self.trajectory_bound_points,state=state1)
        xyz_ref = ref[:, :3]
        self.trajectory_xyz_min = xyz_ref.min(axis=0) - self.trajectory_bound_margin
        self.trajectory_xyz_max = xyz_ref.max(axis=0) + self.trajectory_bound_margin
    
    def get_trejectory(self, N=2000, state=None):
        if state is not None:
            state = np.asarray(state, dtype=np.float32)
            if state.shape != (self.state_dim,):
                raise ValueError(
                    f"Expected state with shape ({self.state_dim},), got {state.shape}."
                )
            self.trajectory_start_position = state[:3].copy()
            #self._update_trajectory_bounds()
        
        t_mpc_array = np.arange(N, dtype=np.float32) * self.dt
        ref = np.array([self.trajectory_function(t_mpc) for t_mpc in t_mpc_array], dtype=np.float32)
        return ref
    
    @staticmethod
    def get_trakectory_list():
        trej_list = ['horizontal_circle','vertical_circle','tilted_circle','lemniscate','helix','up','tilted_square','v_square','h_square','point']
        trej_list = ['vertical_circle','horizontal_circle','lemniscate','helix','v_square']

        return trej_list
    @staticmethod
    def get_test_trajectory_list():
        ############correct this
        trej_list = ['tilted_square','h_spiral','v_figure8','h_figure8','tilted_circle']#'tilted_figure8','lemniscate','helix','up','v_square','h_square','point']
   
        return trej_list
    @staticmethod
    def get_full_trajectory_list():
        trej_list = ['horizontal_circle','vertical_circle','tilted_circle','lemniscate','helix','up','tilted_square','v_square','h_square','point']
        ############correct this
        return ['tilted_figure8','v_spiral','h_sprial']
    
    def get_obstacle_position(self):
    
        ob1 = [0.5,-0.05,2,0.07]
        ob2 = [1.03,0.8,2,0.05]
        ob3 = [-0.1,0.02,0.6,0.03]
        ob4= [-0.9,0.3,1.2,0.05] 
        return [ob1,ob2,ob3,ob4]
    
    def trajectory_function(self, t):
        if t <  self.hover_begin:
            pxr = self.trajectory_start_position[0]
            pyr = self.trajectory_start_position[1]
            pzr = self.trajectory_start_position[2]
            vzr = 0.0
            vxr = 0.0
            vyr = 0.0
            
        else:
            
            t -= self.hover_begin
            if self.trajectory_type == 'horizontal_circle':
                a = 1.0
                omega = 0.5 * np.tanh(0.1 * t)
                pxr = self.trajectory_start_position[0] + a * np.cos(omega * t) - a
                pyr = self.trajectory_start_position[1] + a * np.sin(omega * t)
                pzr = self.trajectory_start_position[2]
                vxr = -a * omega * np.sin(omega * t)
                vyr = a * omega * np.cos(omega * t)
                vzr = 0.0
            elif self.trajectory_type == 'vertical_circle':
                a = 1.0
                omega = 0.5 * np.tanh(0.1 * t)
                pxr = self.trajectory_start_position[0] + a * np.sin(-omega * t + np.pi)
                pyr = self.trajectory_start_position[1]
                pzr = self.trajectory_start_position[2] + a * np.cos(-omega * t + np.pi) + a
                vxr = -a * omega * np.cos(-omega * t + np.pi)
                vyr = 0.0
                vzr = a * omega * np.sin(-omega * t + np.pi)
            elif self.trajectory_type == 'tilted_circle':
                a = 0.5
                c = 0.3
                omega = 0.5 * np.tanh(0.1 * t)
                pxr = self.trajectory_start_position[0] + a * np.cos(omega * t) - a
                pyr = self.trajectory_start_position[1] + a * np.sin(omega * t)
                pzr = self.trajectory_start_position[2] + c * np.sin(omega * t)
                vxr = -a * omega * np.sin(omega * t)
                vyr = a * omega * np.cos(omega * t)
                vzr = c * omega * np.cos(omega * t)
            elif self.trajectory_type == 'lemniscate':
                a = 1.0
                b = 0.5 * np.tanh(0.1 * t)
                pxr = self.trajectory_start_position[0] + a * np.sin(b * t)
                pyr = self.trajectory_start_position[1] + a * np.sin(b * t) * np.cos(b * t)
                pzr = self.trajectory_start_position[2]
                vxr = a * b * np.cos(b * t)
                vyr = a * b * np.cos(2 * b * t)
                vzr = 0.0
            elif self.trajectory_type == 'helix':
                a = 1.0
                T_end = 10.0
                helix_velocity = 0.2
                omega = 0.5 * np.tanh(0.1 * t)
                pxr = self.trajectory_start_position[0] + a * np.cos(omega * t) - a
                pyr = self.trajectory_start_position[1] + a * np.sin(omega * t)
                vxr = -a * omega * np.sin(omega * t)
                vyr = a * omega * np.cos(omega * t)
                if t < T_end:
                    pzr = self.trajectory_start_position[2] + helix_velocity * t
                    vzr = helix_velocity
                else:
                    pzr = self.trajectory_start_position[2] + helix_velocity * T_end
                    vzr = 0.0
            elif self.trajectory_type == 'up':
                climb_velocity = 0.2
                climb_end_time = 6.0
                pxr = self.trajectory_start_position[0]
                pyr = self.trajectory_start_position[1]
                if t < climb_end_time:
                    pzr = self.trajectory_start_position[2] + climb_velocity * t
                    vzr = climb_velocity
                else:
                    pzr = self.trajectory_start_position[2] + climb_velocity * climb_end_time
                    vzr = 0.0
                vxr = 0.0
                vyr = 0.0
            elif self.trajectory_type == 'h_square':
                side_length = 1.0
                speed = 0.2
                pos_uv, vel_uv = self._square_progress(t, side_length, speed)
                pxr = self.trajectory_start_position[0] + pos_uv[0]
                pyr = self.trajectory_start_position[1] + pos_uv[1]
                pzr = self.trajectory_start_position[2]
                vxr = vel_uv[0]
                vyr = vel_uv[1]
                vzr = 0.0
            elif self.trajectory_type == 'v_square':
                side_length = 1.0
                speed = 0.2
                pos_uv, vel_uv = self._square_progress(t, side_length, speed)
                pxr = self.trajectory_start_position[0] + pos_uv[0]
                pyr = self.trajectory_start_position[1]
                pzr = self.trajectory_start_position[2] + pos_uv[1]
                vxr = vel_uv[0]
                vyr = 0.0
                vzr = vel_uv[1]
            elif self.trajectory_type == 'tilted_square':
                side_length = 1.0
                speed = 0.2
                pos_uv, vel_uv = self._square_progress(t, side_length, speed)
                dir_1 = np.array([1.0, 0.0, 0.0], dtype=np.float32)
                dir_2 = np.array([0.0, 1.0, 0.6], dtype=np.float32)
                dir_2 = dir_2 / np.linalg.norm(dir_2)
                offset = pos_uv[0] * dir_1 + pos_uv[1] * dir_2
                velocity = vel_uv[0] * dir_1 + vel_uv[1] * dir_2
                pxr = self.trajectory_start_position[0] + offset[0]
                pyr = self.trajectory_start_position[1] + offset[1]
                pzr = self.trajectory_start_position[2] + offset[2]
                vxr = velocity[0]
                vyr = velocity[1]
                vzr = velocity[2]
            elif self.trajectory_type == 'tilted_figure8':
                a = 0.5
                c = 0.2
                omega = 0.4 * np.tanh(0.1 * t)
                s = omega * t
                pxr = self.trajectory_start_position[0] + a * np.sin(s)
                pyr = self.trajectory_start_position[1] + 0.5 * a * np.sin(2.0 * s)
                pzr = self.trajectory_start_position[2] + c * np.sin(s)
                vxr = a * omega * np.cos(s)
                vyr = a * omega * np.cos(2.0 * s)
                vzr = c * omega * np.cos(s)
            elif self.trajectory_type == 'v_figure8':
                a = 0.5
                c = 0.2
                omega = 0.4 * np.tanh(0.1 * t)
                s = omega * t
                pxr = self.trajectory_start_position[0] - 0.5 * c * np.sin(2.0 * s)
                pyr = self.trajectory_start_position[1]
                pzr = self.trajectory_start_position[2] + a * np.sin(s)
                vxr = -c * omega * np.cos(2.0 * s)
                vyr = 0.0
                vzr = a * omega * np.cos(s)
            elif self.trajectory_type == 'h_spiral2':
                r = 0.5
                omega = 0.4 * np.tanh(0.1 * t)
                climb_velocity = 0.02
                s = omega * t
                pxr = self.trajectory_start_position[0] + r * np.cos(s) - r
                pyr = self.trajectory_start_position[1]
                pzr = self.trajectory_start_position[2] + r * np.sin(s) + climb_velocity * t
                vxr = -r * omega * np.sin(s)
                vyr = 0.0
                vzr = r * omega * np.cos(s) + climb_velocity
            elif self.trajectory_type == 'h_spiral':
                growth = 0.05
                omega = 0.6 * np.tanh(0.1 * t)
                r = min(0.5, growth * t)
                s = omega * t
                pxr = self.trajectory_start_position[0] + r * np.cos(s)
                pyr = self.trajectory_start_position[1] + r * np.sin(s)
                pzr = self.trajectory_start_position[2]
                vxr = growth * np.cos(s) - r * omega * np.sin(s)
                vyr = growth * np.sin(s) + r * omega * np.cos(s)
                vzr = 0.0
            elif self.trajectory_type == 'point':
                climb_velocity = 0.02
                climb_end_time = 0.0
                pxr = self.trajectory_start_position[0]
                pyr = self.trajectory_start_position[1]
                pzr = self.trajectory_start_position[2]
                # if t < climb_end_time:
                #     pzr = self.trajectory_start_position[2] + climb_velocity * t
                #     vzr = climb_velocity
                # else:
                #     pzr = self.trajectory_start_position[2] + climb_velocity * climb_end_time
                #     vzr = 0.0
                vzr = 0.0
                vxr = 0.0
                vyr = 0.0
            elif self.trajectory_type == "sin":
                speed = 0.2
                amplitude = 0.45        # How wide/deep the drone weaves
                cycle_length = 1.4      # The physical distance of one full sine wave
                num_cycles = 2           # 2 full cycles per plane
                hover_time = 1
                total_progress = num_cycles * cycle_length
                # Calculate how long it takes to fly the first set of cycles
                phase_time = total_progress / speed
                
                base_x = self.trajectory_start_position[0]
                base_y = self.trajectory_start_position[1]
                base_z = self.trajectory_start_position[2]

                def smooth_progress(time_in_phase):
                    u = np.clip(time_in_phase / phase_time, 0.0, 1.0)
                    s = u * u * u * (u * (u * 6.0 - 15.0) + 10.0)
                    ds_du = 30.0 * u * u * (u - 1.0) * (u - 1.0)
                    progress = total_progress * s
                    progress_rate = (total_progress / phase_time) * ds_du
                    return progress, progress_rate
                
                # Wave frequency multiplier (2*pi / wavelength)
                k = 2.0 * np.pi / cycle_length

                if t < phase_time:
                    # --- Phase 1: X-Y Plane (y = sin(x)) ---
                    path_progress, progress_rate = smooth_progress(t)
                    pxr = base_x + path_progress
                    pyr = base_y + amplitude * np.sin(k * path_progress)
                    pzr = base_z
                    
                    # Velocity is the derivative of position
                    vxr = progress_rate
                    vyr = amplitude * k * progress_rate * np.cos(k * path_progress)
                    vzr = 0.0
                elif t < phase_time + hover_time:
                    pxr = base_x + total_progress
                    pyr = base_y
                    pzr = base_z 
                    vzr = 0.0
                    vxr = 0.0
                    vyr = 0.0
                else:
                    time_in_phase_2 = t - phase_time - hover_time
                    z_progress, progress_rate = smooth_progress(time_in_phase_2)
                    
                    # The exact X position the drone finished at after Phase 1
                    x_end_of_phase_1 = base_x + total_progress
                    
                    pxr = x_end_of_phase_1 + amplitude * np.sin(k * z_progress)
                    pyr = base_y
                    pzr = base_z + z_progress
                    
                    # Velocity
                    vxr = amplitude * k * progress_rate * np.cos(k * z_progress)
                    vyr = 0.0
                    vzr = progress_rate
            else:
                raise ValueError(f"Unknown trajectory_type: {self.trajectory_type}")
        return np.array([pxr, pyr, pzr, vxr, vyr, vzr, 0.0, 0.0, 0.0], dtype=np.float32)

    def reset(self, obs=None, seed=None):
        if seed is not None:
            np.random.seed(seed)
        if obs is None:
            self.state = np.zeros(self.state_dim, dtype=np.float32)
        else:
            obs = np.asarray(obs, dtype=np.float32)
            if obs.shape != (self.state_dim,):
                raise ValueError(
                    f"Expected obs with shape ({self.state_dim},), got {obs.shape}."
                )
            self.state = obs.copy()
        self.trajectory_start_position = self.state[:3].copy()
        self._update_trajectory_bounds()

        self.obs = self.state.copy()
        self.steps = 0
        return self.obs.copy(), {}

    def step(self, action):
        if action is None:
            return self.obs.copy(), 0.0, True, {}
        action = np.asarray(action, dtype=np.float32)
        if action.shape != (self.action_dim,):
            raise ValueError(
                f"Expected action with shape ({self.action_dim},), got {action.shape}."
            )
        #action = self.clip_action(action)
        
        next_state = self.dynamics(self.state, action).astype(np.float32)

        self.state = next_state
        self.obs = self.state.copy()
        self.steps += 1

        roll = float(self.state[6])
        pitch = float(self.state[7])
        yaw = self._wrap_to_pi(float(self.state[8]))

        attitude_fall = (
            abs(roll) > self.max_roll_pitch_rad
            or abs(pitch) > self.max_roll_pitch_rad
            or abs(yaw) > self.max_yaw_rad
            or self.state[2] < 0
        )
        xyz = self.state[:3]
        if self.trajectory_type == 'inf':
            out_of_trajectory_bounds = False
        else:
            out_of_trajectory_bounds = bool(
            np.any(xyz < self.trajectory_xyz_min) or np.any(xyz > self.trajectory_xyz_max)
             )

        done = attitude_fall or out_of_trajectory_bounds
        if self.max_episode_steps is not None and self.steps >= self.max_episode_steps:
            done = True

        reward = 0.0
        info = {
            "attitude_fall": attitude_fall,
            "out_of_trajectory_bounds": out_of_trajectory_bounds,
            "roll": roll,
            "pitch": pitch,
            "yaw": yaw,
        }
        return self.obs.copy(), reward, done, info

    def dynamics_torch(self, state, action):
        """
        PyTorch-compatible dynamics for MPPI.
        
        Args:
            state: Tensor of shape (B, 9) [px, py, pz, vx, vy, vz, roll, pitch, yaw]
            action: Tensor of shape (B, 4) [roll_c, pitch_c, yaw_c, thrust]
            dt: Time step (float)
            
        Returns:
            next_state: Tensor of shape (B, 9)
        """
        # Physics constants
        m = self.mass
        g = self.gravity
        tau = self.tau

        dt = self.dt
        # Unpack state (B, 9)
        px, py, pz = state[:, 0], state[:, 1], state[:, 2]
        vx, vy, vz = state[:, 3], state[:, 4], state[:, 5]
        roll, pitch, yaw = state[:, 6], state[:, 7], state[:, 8]

        # Unpack action (B, 4)
        action[:,3] += 0.028*g
        action = self.clip_action_torch(action)
        roll_c = action[:, 0]
        pitch_c = action[:, 1]
        yaw_c = action[:, 2]
        thrust = action[:, 3]
        thrust_eff = thrust * self._ground_effect_multiplier_torch(pz)

     
        # Precompute trigonometric terms
        cp = torch.cos(pitch)
        sp = torch.sin(pitch)
        cr = torch.cos(roll)
        sr = torch.sin(roll)
        cy = torch.cos(yaw)
        sy = torch.sin(yaw)

        # Rotation Matrix (Body Z-axis column only, for Thrust)
        # R13 = cr * sp * cy + sr * sy
        # R23 = cr * sp * sy - sr * cy
        # R33 = cr * cp
        
        # Calculate Accelerations
        # vdot = [0,0,-g] + R @ [0,0,T] / m
        ax = ((cr * sp * cy + sr * sy) * thrust_eff) / m
        ay = ((cr * sp * sy - sr * cy) * thrust_eff) / m
        az = ((cr * cp) * thrust_eff) / m - g

        # Euler Integration
        px_new = px + vx * dt
        py_new = py + vy * dt
        pz_new = pz + vz * dt

        vx_new = vx + ax * dt
        vy_new = vy + ay * dt
        vz_new = vz + az * dt

        # Attitude First-Order Filter
        roll_new = roll + (roll_c - roll) / tau * dt
        pitch_new = pitch + (pitch_c - pitch) / tau * dt
        yaw_new = yaw + (yaw_c - yaw) / tau * dt

        # Stack into next state vector
        next_state = torch.stack(
            [px_new, py_new, pz_new, vx_new, vy_new, vz_new, roll_new, pitch_new, yaw_new], 
            dim=1
        )
        
        return next_state
    def dynamics(self, x, u):
        dt = self.dt
        """
        Numpy dynamics for a single state/action sample.
        x: (9,) [px, py, pz, vx, vy, vz, roll, pitch, yaw]
        u: (4,) [roll_c, pitch_c, yaw_c, thrust]
        Returns: next_x (9,)
        """
        x = np.asarray(x, dtype=np.float32)
        u = np.asarray(u, dtype=np.float32)
        if x.shape != (self.state_dim,):
            raise ValueError(f"Expected x with shape ({self.state_dim},), got {x.shape}.")
        if u.shape != (self.action_dim,):
            raise ValueError(f"Expected u with shape ({self.action_dim},), got {u.shape}.")
        u[3] += self.gravity * 0.028
     #   print(self.mass)
        u = self.clip_action(u)

        # Extract states
        px, py, pz = x[0], x[1], x[2]
        vx, vy, vz = x[3], x[4], x[5]
        roll, pitch, yaw = x[6], x[7], x[8]

        # Extract controls
        roll_c, pitch_c, yaw_c, thrust = u[0], u[1], u[2], u[3]
        thrust_eff = thrust * self._ground_effect_multiplier(pz)

        # Rotation Matrix calculation
        cp = np.cos(pitch)
        sp = np.sin(pitch)
        cr = np.cos(roll)
        sr = np.sin(roll)
        cy = np.cos(yaw)
        sy = np.sin(yaw)

        # We only need the column relevant to Thrust (Z-axis of body frame)
        # R * [0, 0, T] means we only need the 3rd column of R
        R13 = cr * sp * cy + sr * sy
        R23 = cr * sp * sy - sr * cy
        R33 = cr * cp

        # Accelerations
        # vdot = [0,0,-g] + R @ [0,0,T]/m
        ax = (R13 * thrust_eff) / self.mass
        ay = (R23 * thrust_eff) / self.mass
        az = -self.gravity + (R33 * thrust_eff) / self.mass

        # Euler Integration
        # Position
        px_new = px + vx * dt
        py_new = py + vy * dt
        pz_new = pz + vz * dt

        # Velocity
        vx_new = vx + ax * dt
        vy_new = vy + ay * dt
        vz_new = vz + az * dt

        # Attitude (First order filter dynamics as defined in your model)
        roll_new = roll + ((roll_c - roll) / self.tau) * dt
        pitch_new = pitch + ((pitch_c - pitch) / self.tau) * dt
        yaw_new = yaw + ((yaw_c - yaw) / self.tau) * dt

        return np.array(
            [px_new, py_new, pz_new, vx_new, vy_new, vz_new, roll_new, pitch_new, yaw_new],
            dtype=np.float32,
        )
    
