import numpy as np
import gymnasium as gym
from gymnasium.envs.mujoco import MujocoEnv 
import mujoco
from gymnasium import spaces
import os 
import collections

class S0100Env(MujocoEnv):
    metadata = {"render_modes": ["human", "rgb_array", "depth_array"]}
    
    def __init__(self, render_mode=None, max_episode_steps=200, success_distance=0.03, success_hold_steps=3, obs_type="blind"):
        xml_file = os.path.join(os.path.dirname(__file__), "SO101/block_stack.xml")
        frame_skip = 20
        
        self.img_height = 84
        self.img_width = 84
        self.n_actuators = 6
        self.joint_obs_dim = 39
        
        self.obs_type = obs_type 

        self.observation_space = spaces.Dict({
            "joints": spaces.Box(low=-np.inf, high=np.inf, shape=(self.joint_obs_dim,), dtype=np.float32),
            "image": spaces.Box(low=0, high=255, shape=(self.img_height, self.img_width, 3), dtype=np.uint8)
        })
        
        # Action space: Outputs values between -1 and 1 and will be saled to actual robot limits
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(self.n_actuators,), dtype=np.float32)

        self.max_episode_steps = int(max_episode_steps)
        self.success_hold_steps = int(success_hold_steps) 
        self.target_tolerance = success_distance          
        self._episode_step = 0
        self._success_streak = 0
        self._prev_action = np.zeros(self.n_actuators, dtype=np.float32)
        
        super().__init__(model_path=xml_file, frame_skip=frame_skip, observation_space=self.observation_space, render_mode=render_mode)
        self.mujoco_renderer = mujoco.Renderer(self.model, height=self.img_height, width=self.img_width)
        self.tip_site_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, "gripperframe")
        self.red_block_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, "block_red_site")
        self.blue_block_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, "block_blue_site")
        self.target_bottom_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, "target_bottom_site")
        self.target_top_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, "target_top_site")

        self.red_joint_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, "block_red_joint")
        self.blue_joint_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, "block_blue_joint")

    def _scale_action(self, action):
        action = np.asarray(action, dtype=np.float32)
        action = np.clip(action, -1.0, 1.0)
        ctrl_min = self.model.actuator_ctrlrange[:, 0]
        ctrl_max = self.model.actuator_ctrlrange[:, 1]
        action_scaled = ctrl_min + (action + 1.0) * 0.5 * (ctrl_max - ctrl_min)
        return np.clip(action_scaled, ctrl_min, ctrl_max)
    
    def compute_reward(self, gripper_pos, red_block_pos, blue_block_pos, target_bottom_pos, target_top_pos, action):
        # Calculate distances based purely on historical buffer data or HER injected goals
        dist_grab_red = np.linalg.norm(gripper_pos - red_block_pos)
        dist_place_red = np.linalg.norm(red_block_pos - target_bottom_pos)
        dist_grab_blue = np.linalg.norm(gripper_pos - blue_block_pos)
        dist_stack_blue = np.linalg.norm(blue_block_pos - target_top_pos)

        reward = 0.0
        is_success = False
        grab_tolerance = 0.02
        target_tol = self.target_tolerance 

        # The Red Block (Bottom)
        if dist_place_red > target_tol:
            reward -= (5.0 * dist_grab_red)
            if dist_grab_red < grab_tolerance:
                reward -= (10.0 * dist_place_red)
                reward += 1.0 

        # The Blue Block (Top) 
        else:
            reward += 5.0 
            reward -= (5.0 * dist_grab_blue)
            
            if dist_grab_blue < grab_tolerance:
                reward -= (10.0 * dist_stack_blue)
                reward += 1.0 

        # Task Completion Evaluation
        if dist_place_red < target_tol and dist_stack_blue < target_tol:
            reward += 50.0 
            is_success = True

        action_penalty = np.sum(np.square(action)) * 0.001
        reward -= action_penalty

        return reward / 10.0, is_success
        
    def step(self, action):
        # What happens during one time step
        self._episode_step += 1
        # Scale action so it matches arm limits
        action_scaled = self._scale_action(action)
        
        # Do simulation does a lot of the stuff of moving the robot
        self.do_simulation(action_scaled, self.frame_skip)
        observation = self._get_obs()

        gripper_pos = self.data.site_xpos[self.tip_site_id]
        red_block_pos = self.data.site_xpos[self.red_block_id]
        blue_block_pos = self.data.site_xpos[self.blue_block_id]
        target_bottom_pos = self.data.site_xpos[self.target_bottom_id]
        target_top_pos = self.data.site_xpos[self.target_top_id]

        reward, is_success = self.compute_reward(gripper_pos=gripper_pos, red_block_pos=red_block_pos, blue_block_pos=blue_block_pos, target_bottom_pos=target_bottom_pos, target_top_pos=target_top_pos)

        if is_success:
            self._success_streak += 1
        else:
            self._success_streak = 0

        # Terminated: Agent achieved the goal and held it
        terminated = bool(self._success_streak >= self.success_hold_steps)
        truncated = bool(self._episode_step >= self.max_episode_steps and not terminated)
        # Terminated: For if the robot carried out outcome or note
        # Trunated: If the robot action neither succeeded or failed (usually a time limit)
        
        # fix this
        info = {
            "is_success": is_success,
            "success_streak": self._success_streak,
            "distances": self.get_distance() # Handy for logging/debugging
        }
        if self.render_mode == "human":
            self.render()
        
        return observation, reward, terminated, truncated, info
    
    def reset_model(self):
        mujoco.mj_resetData(self.model, self.data)

        # Add a tiny bit of uniform noise to the starting joint angles
        noise_low = -0.01
        noise_high = 0.01
        
        # Copy the initial states so we don't permanently alter them
        qpos = self.init_qpos.copy()
        qvel = self.init_qvel.copy()

        # Add noise to the first 6 indices (the arm joints)
        qpos[:6] = qpos[:6] + self.np_random.uniform(low=noise_low, high=noise_high, size=self.n_actuators)
        # qvel[:6] = qvel[:6] + self.np_random.uniform(low=-0.005, high=0.005, size=self.model.nv)
        
        block_noise_range = 0.02

        red_qpos_idx = self.model.jnt_qposadr[self.red_block_id]
        blue_qpos_idx = self.model.jnt_qposadr[self.blue_block_id]

        qpos[red_qpos_idx : red_qpos_idx + 2] += self.np_random.uniform(low=-block_noise_range, high=block_noise_range, size=2)
        qpos[blue_qpos_idx : blue_qpos_idx + 2] += self.np_random.uniform( low=-block_noise_range, high=block_noise_range, size=2)

        # Set the state with the noisy positions
        self.set_state(qpos, qvel)
        mujoco.mj_forward(self.model, self.data)

        self._episode_step = 0
        self._success_streak = 0
        self._prev_action = np.zeros(self.n_actuators, dtype=np.float32)

        return self._get_obs()
    
    
    def _get_obs(self):
        joint_positions = self.data.qpos[:self.n_actuators].copy()
        joint_velocities = self.data.qvel[:self.n_actuators].copy()
        tip_pos = self.data.site_xpos[self.tip_site_id].copy()
        red_block_pos = self.data.site_xpos[self.red_block_id].copy()
        blue_block_pos = self.data.site_xpos[self.blue_block_id].copy()
        target_bottom_pos = self.data.site_xpos[self.target_bottom_id].copy()
        target_top_pos = self.data.site_xpos[self.target_top_id].copy()

        vec_gripper_to_red = red_block_pos - tip_pos
        vec_red_to_target = target_bottom_pos - red_block_pos
        vec_gripper_to_blue = blue_block_pos - tip_pos
        vec_blue_to_target = target_top_pos - blue_block_pos

        joint_obs = np.concatenate([
            joint_positions,             
            joint_velocities,                                       
            vec_gripper_to_red,   
            vec_red_to_target,    
            vec_gripper_to_blue,  
            vec_blue_to_target,
            tip_pos, 
            red_block_pos,        
            blue_block_pos, 
            # My goals for HER
            target_bottom_pos,   
            target_top_pos
        ]).astype(np.float32)

        if self.obs_type == "state":
            return joint_obs


        self.mujoco_renderer.update_scene(self.data, camera="gripper_camera")
        rgb_image = self.mujoco_renderer.render()

        if self.obs_type == "blind":
            rgb_image = np.zeros((self.img_height, self.img_width, 3), dtype=np.uint8)
        else:
            self.mujoco_renderer.update_scene(self.data, camera="gripper_camera")
            rgb_image = self.mujoco_renderer.render()

        return {
            "joints": joint_obs,
            "image": rgb_image
        }
        
    def close(self):
        # Manually close the renderer to prevent those "NoneType" errors
        if self.mujoco_renderer:
            self.mujoco_renderer.close()
        super().close()


# Using to stack last 3 frames so robot knows what it's seeing
class FrameStack(gym.ObservationWrapper):
    def __init__(self, env, num_stack=3):
        super().__init__(env)
        self.num_stack = num_stack
        self.frames = collections.deque(maxlen=num_stack)
        obs_space = env.observation_space
        old_image_shape = obs_space["image"].shape # (84, 84, 3)
        
        # New shape will be (84, 84, 9) 
        new_image_shape = (old_image_shape[0], old_image_shape[1], old_image_shape[2] * num_stack)
        
        self.observation_space = spaces.Dict({
            "joints": obs_space["joints"],
            "image": spaces.Box(low=0, high=255, shape=new_image_shape, dtype=np.uint8)
        })

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        for i in range(self.num_stack):
            self.frames.append(obs["image"])
            
        return self.observation(obs), info

    def observation(self, obs):
        self.frames.append(obs["image"])
        
        obs["image"] = np.concatenate(list(self.frames), axis=-1)
        
        return obs