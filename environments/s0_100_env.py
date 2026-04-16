# import numpy as np
# import gymnasium as gym
# from gymnasium.envs.mujoco import MujocoEnv 
# import mujoco
# from gymnasium import spaces
# import os 

# class S0100Env(MujocoEnv):
#     metadata = {"render_modes": ["human", "rgb_array", "depth_array"]}
    
#     def __init__(self, render_mode=None, max_episode_steps=200, success_distance=0.03, success_hold_steps=3):
#         xml_file = os.path.join(os.path.dirname(__file__), "SO101/so101_new_calib.xml")
#         frame_skip = 20
        
#         self.img_height = 84
#         self.img_width = 84
#         self.n_actuators = 6
#         self.joint_obs_dim = 18
        
#         self.observation_space = spaces.Dict({
#             "joints": spaces.Box(low=-np.inf, high=np.inf, shape=(self.joint_obs_dim,), dtype=np.float32),
#             "image": spaces.Box(low=0, high=255, shape=(self.img_height, self.img_width, 3), dtype=np.uint8)
#         })
        
#         # Action space: Outputs values between -1 and 1 and will be saled to actual robot limits
#         self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(self.n_actuators,), dtype=np.float32)
#         self.init_target_pos = np.array([0.25, 0.0, 0.15], dtype=np.float64)
#         self.target_pos = self.init_target_pos.copy()
#         self.target_low = np.array([0.15, -0.18, 0.08], dtype=np.float64)
#         self.target_high = np.array([0.35, 0.18, 0.24], dtype=np.float64)

#         self.max_episode_steps = int(max_episode_steps)
#         self.success_distance = float(success_distance)
#         self.success_hold_steps = int(success_hold_steps)
#         self._episode_step = 0
#         self._success_streak = 0
#         self._prev_distance = None
#         self._prev_action = np.zeros(self.n_actuators, dtype=np.float32)
        
#         super().__init__(model_path=xml_file, frame_skip=frame_skip, observation_space=self.observation_space, render_mode=render_mode)
#         self.mujoco_renderer = mujoco.Renderer(self.model, height=self.img_height, width=self.img_width)
#         self.tip_site_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, "gripperframe")
#         self.target_body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "target")
#         self.target_mocap_id = int(self.model.body_mocapid[self.target_body_id])

#         if self.target_mocap_id < 0:
#             raise ValueError("The target body must be configured as a mocap body in the model XML.")

#     def _scale_action(self, action):
#         action = np.asarray(action, dtype=np.float32)
#         action = np.clip(action, -1.0, 1.0)
#         ctrl_min = self.model.actuator_ctrlrange[:, 0]
#         ctrl_max = self.model.actuator_ctrlrange[:, 1]
#         action_scaled = ctrl_min + (action + 1.0) * 0.5 * (ctrl_max - ctrl_min)
#         return np.clip(action_scaled, ctrl_min, ctrl_max)
        
#     def step(self, action):
#         # What happens during one time step
#         self._episode_step += 1
        
#         # Scale action so it matches arm limits
#         action_scaled = self._scale_action(action)
        
#         # Do simulation does a lot of the stuff of moving the robot
#         self.do_simulation(action_scaled, self.frame_skip)
#         observation = self._get_obs() # Gets observations from class function
        
#         # Now we calculate the reward from each step

#         tip_pos = self.data.site_xpos[self.tip_site_id].copy()
#         distance = np.linalg.norm(tip_pos - self.target_pos)

#         progress = 0.0 if self._prev_distance is None else (self._prev_distance - distance)
#         action = np.asarray(action, dtype=np.float32)
#         action_cost = 0.01 * np.mean(np.square(action))
#         smooth_cost = 0.005 * np.mean(np.square(action - self._prev_action))

#         success_now = bool(distance < self.success_distance)
#         if success_now:
#             self._success_streak += 1
#         else:
#             self._success_streak = 0

#         reward_components = {
#             "distance": -distance,
#             "progress": 2.0 * progress,
#             "action_cost": -action_cost,
#             "smooth_cost": -smooth_cost,
#             "success_bonus": 1.0 if success_now else 0.0,
#         }
#         reward = float(sum(reward_components.values()))

#         terminated = bool(self._success_streak >= self.success_hold_steps)
#         truncated = bool(self._episode_step >= self.max_episode_steps and not terminated)
#         # Terminated: For if the robot carried out outcome or note
#         # Trunated: If the robot action neither succeeded or failed (usually a time limit)

#         self._prev_distance = float(distance)
#         self._prev_action = action.copy()
        
#         info = {
#             "distance": float(distance),
#             "tip_pos": tip_pos,
#             "target_pos": self.target_pos.copy(),
#             "success": terminated,
#             "timeout": truncated,
#             "success_streak": self._success_streak,
#             "reward_components": reward_components,
#         }
#         if self.render_mode == "human":
#             self.render()
        
#         return observation, reward, terminated, truncated, info
        
#     def reset_model(self):
#         # Where does the model start each time
        
#         # noise so it doesn't start in the exact same place
#         noise_low = -0.01
#         noise_high = 0.01
        
#         qpos = self.init_qpos + self.np_random.uniform(low=noise_low, high=noise_high, size=self.model.nq)
#         qvel = self.init_qvel + self.np_random.uniform(low=-0.005, high=0.005, size=self.model.nv)
#         self.target_pos = self.np_random.uniform(low=self.target_low, high=self.target_high, size=3)
#         self.set_state(qpos, qvel)

#         self.data.mocap_pos[self.target_mocap_id] = self.target_pos
#         self.data.mocap_quat[self.target_mocap_id] = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64)
#         mujoco.mj_forward(self.model, self.data)

#         tip_pos = self.data.site_xpos[self.tip_site_id].copy()
#         distance = float(np.linalg.norm(tip_pos - self.target_pos))
#         if not np.isfinite(distance):
#             raise RuntimeError("Invalid environment state: non-finite target distance after reset.")

#         self._episode_step = 0
#         self._success_streak = 0
#         self._prev_distance = distance
#         self._prev_action = np.zeros(self.n_actuators, dtype=np.float32)
        
#         return self._get_obs()
    
#     def _get_obs(self):
#         joint_positions = self.data.qpos[:self.n_actuators].copy()
#         joint_velocities = self.data.qvel[:self.n_actuators].copy()
#         tip_pos = self.data.site_xpos[self.tip_site_id].copy()
#         joint_obs = np.concatenate([joint_positions, joint_velocities, tip_pos, self.target_pos]).astype(np.float32)

#         self.mujoco_renderer.update_scene(self.data, camera="gripper_camera")
#         rgb_image = self.mujoco_renderer.render()

#         return {
#             "joints": joint_obs,
#             "image": rgb_image
#         }
        
#     def close(self):
#         # Manually close the renderer to prevent those "NoneType" errors
#         if self.mujoco_renderer:
#             self.mujoco_renderer.close()
#         super().close()

import numpy as np
import gymnasium as gym
from gymnasium.envs.mujoco import MujocoEnv 
import mujoco
from gymnasium import spaces
import os 

class S0100Env(MujocoEnv):
    metadata = {"render_modes": ["human", "rgb_array", "depth_array"]}
    
    def __init__(self, render_mode=None, max_episode_steps=200, success_distance=0.03, success_hold_steps=3):
        xml_file = os.path.join(os.path.dirname(__file__), "SO101/so101_new_calib.xml")
        frame_skip = 20
        
        self.img_height = 84
        self.img_width = 84
        self.n_actuators = 6
        self.joint_obs_dim = 18
        
        self.observation_space = spaces.Dict({
            "joints": spaces.Box(low=-np.inf, high=np.inf, shape=(self.joint_obs_dim,), dtype=np.float32),
            "image": spaces.Box(low=0, high=255, shape=(self.img_height, self.img_width, 3), dtype=np.uint8)
        })
        
        # Action space: Outputs values between -1 and 1 and will be saled to actual robot limits
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(self.n_actuators,), dtype=np.float32)
        self.init_target_pos = np.array([0.25, 0.0, 0.15], dtype=np.float64)
        self.target_pos = self.init_target_pos.copy()
        self.target_low = np.array([0.18, -0.12, 0.10], dtype=np.float64)
        self.target_high = np.array([0.32, 0.12, 0.22], dtype=np.float64)
        self.target_visibility_attempts = 40
        self.target_min_red_pixels = 2
        self.target_center_margin = 0.98

        self.max_episode_steps = int(max_episode_steps)
        self.success_distance = float(success_distance)
        self.success_hold_steps = int(success_hold_steps)
        self._episode_step = 0
        self._success_streak = 0
        self._prev_distance = None
        self._prev_action = np.zeros(self.n_actuators, dtype=np.float32)
        self._prev_action_delta = np.zeros(self.n_actuators, dtype=np.float32)
        self._last_target_visible = False
        self._last_target_pixel_count = 0
        self._last_target_center_offset = np.nan
        
        super().__init__(model_path=xml_file, frame_skip=frame_skip, observation_space=self.observation_space, render_mode=render_mode)
        self.mujoco_renderer = mujoco.Renderer(self.model, height=self.img_height, width=self.img_width)
        self.tip_site_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, "gripperframe")
        self.target_body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "target")
        self.target_mocap_id = int(self.model.body_mocapid[self.target_body_id])

        if self.target_mocap_id < 0:
            raise ValueError("The target body must be configured as a mocap body in the model XML.")

    def _target_visibility_from_frame(self, frame):
        if frame is None:
            return False, 0, np.nan

        red = frame[:, :, 0].astype(np.int16)
        green = frame[:, :, 1].astype(np.int16)
        blue = frame[:, :, 2].astype(np.int16)
        mask = (red > 100) & (green < 140) & (blue < 140) & ((red - np.maximum(green, blue)) > 20)

        pixel_count = int(mask.sum())
        if pixel_count < self.target_min_red_pixels:
            return False, pixel_count, np.nan

        ys, xs = np.where(mask)
        cx = float(xs.mean())
        cy = float(ys.mean())
        center_x = (frame.shape[1] - 1) * 0.5
        center_y = (frame.shape[0] - 1) * 0.5
        norm_dx = abs(cx - center_x) / max(1.0, center_x)
        norm_dy = abs(cy - center_y) / max(1.0, center_y)
        center_offset = float(np.sqrt(norm_dx ** 2 + norm_dy ** 2))

        in_view = bool(norm_dx <= self.target_center_margin and norm_dy <= self.target_center_margin)
        return in_view, pixel_count, center_offset

    def _sample_visible_target(self):
        best_pos = self.init_target_pos.copy()
        best_score = -np.inf
        best_visible = False
        best_pixels = 0
        best_offset = np.nan

        for _ in range(self.target_visibility_attempts):
            candidate = self.np_random.uniform(low=self.target_low, high=self.target_high, size=3)
            self.data.mocap_pos[self.target_mocap_id] = candidate
            self.data.mocap_quat[self.target_mocap_id] = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64)
            mujoco.mj_forward(self.model, self.data)

            self.mujoco_renderer.update_scene(self.data, camera="gripper_camera")
            frame = self.mujoco_renderer.render()
            visible, pixels, center_offset = self._target_visibility_from_frame(frame)
            score = 6.0 * float(pixels) - (0.0 if np.isnan(center_offset) else 8.0 * center_offset)

            if score > best_score:
                best_score = score
                best_pos = candidate.copy()
                best_visible = visible
                best_pixels = pixels
                best_offset = center_offset

            if visible:
                return candidate, visible, pixels, center_offset

        return best_pos, best_visible, best_pixels, best_offset

    def _scale_action(self, action):
        action = np.asarray(action, dtype=np.float32)
        action = np.clip(action, -1.0, 1.0)
        ctrl_min = self.model.actuator_ctrlrange[:, 0]
        ctrl_max = self.model.actuator_ctrlrange[:, 1]
        action_scaled = ctrl_min + (action + 1.0) * 0.5 * (ctrl_max - ctrl_min)
        return np.clip(action_scaled, ctrl_min, ctrl_max)
        
    def step(self, action):
        # What happens during one time step
        self._episode_step += 1
        
        # Scale action so it matches arm limits
        action_scaled = self._scale_action(action)
        
        # Do simulation does a lot of the stuff of moving the robot
        self.do_simulation(action_scaled, self.frame_skip)
        observation = self._get_obs() # Gets observations from class function
        
        # Now we calculate the reward from each step

        tip_pos = self.data.site_xpos[self.tip_site_id].copy()
        distance = np.linalg.norm(tip_pos - self.target_pos)

        progress = 0.0 if self._prev_distance is None else (self._prev_distance - distance)
        action = np.asarray(action, dtype=np.float32)
        action_cost = 0.01 * np.mean(np.square(action))
        action_delta = action - self._prev_action
        smooth_cost = 0.03 * np.mean(np.square(action_delta))
        jerk_cost = 0.01 * np.mean(np.square(action_delta - self._prev_action_delta))

        target_visible, target_pixels, target_center_offset = self._target_visibility_from_frame(observation.get("image"))
        self._last_target_visible = target_visible
        self._last_target_pixel_count = target_pixels
        self._last_target_center_offset = target_center_offset

        success_now = bool(distance < self.success_distance)
        if success_now:
            self._success_streak += 1
        else:
            self._success_streak = 0

        reward_components = {
            "distance": -distance,
            "progress": 2.0 * progress,
            "action_cost": -action_cost,
            "smooth_cost": -smooth_cost,
            "jerk_cost": -jerk_cost,
            "success_bonus": 1.0 if success_now else 0.0,
        }
        reward = float(sum(reward_components.values()))

        terminated = bool(self._success_streak >= self.success_hold_steps)
        truncated = bool(self._episode_step >= self.max_episode_steps and not terminated)
        # Terminated: For if the robot carried out outcome or note
        # Trunated: If the robot action neither succeeded or failed (usually a time limit)

        self._prev_distance = float(distance)
        self._prev_action = action.copy()
        self._prev_action_delta = action_delta.copy()
        
        info = {
            "distance": float(distance),
            "tip_pos": tip_pos,
            "target_pos": self.target_pos.copy(),
            "success": terminated,
            "timeout": truncated,
            "success_streak": self._success_streak,
            "target_visible": target_visible,
            "target_pixel_count": int(target_pixels),
            "target_center_offset": float(target_center_offset),
            "action_delta_l2": float(np.linalg.norm(action_delta)),
            "reward_components": reward_components,
        }
        if self.render_mode == "human":
            self.render()
        
        return observation, reward, terminated, truncated, info
        
    def reset_model(self):
        # Where does the model start each time
        
        # noise so it doesn't start in the exact same place
        noise_low = -0.01
        noise_high = 0.01
        
        qpos = self.init_qpos + self.np_random.uniform(low=noise_low, high=noise_high, size=self.model.nq)
        qvel = self.init_qvel + self.np_random.uniform(low=-0.005, high=0.005, size=self.model.nv)
        self.set_state(qpos, qvel)

        self.target_pos, visible, pixels, center_offset = self._sample_visible_target()
        self.data.mocap_pos[self.target_mocap_id] = self.target_pos
        self.data.mocap_quat[self.target_mocap_id] = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64)
        mujoco.mj_forward(self.model, self.data)

        self._last_target_visible = bool(visible)
        self._last_target_pixel_count = int(pixels)
        self._last_target_center_offset = float(center_offset)

        tip_pos = self.data.site_xpos[self.tip_site_id].copy()
        distance = float(np.linalg.norm(tip_pos - self.target_pos))
        if not np.isfinite(distance):
            raise RuntimeError("Invalid environment state: non-finite target distance after reset.")

        self._episode_step = 0
        self._success_streak = 0
        self._prev_distance = distance
        self._prev_action = np.zeros(self.n_actuators, dtype=np.float32)
        self._prev_action_delta = np.zeros(self.n_actuators, dtype=np.float32)
        
        return self._get_obs()
    
    def _get_obs(self):
        joint_positions = self.data.qpos[:self.n_actuators].copy()
        joint_velocities = self.data.qvel[:self.n_actuators].copy()
        tip_pos = self.data.site_xpos[self.tip_site_id].copy()
        joint_obs = np.concatenate([joint_positions, joint_velocities, tip_pos, self.target_pos]).astype(np.float32)

        self.mujoco_renderer.update_scene(self.data, camera="gripper_camera")
        rgb_image = self.mujoco_renderer.render()

        return {
            "joints": joint_obs,
            "image": rgb_image
        }
        
    def close(self):
        if self.mujoco_renderer:
            self.mujoco_renderer.close()
        super().close()