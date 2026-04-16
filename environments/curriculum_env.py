import numpy as np
import gymnasium as gym
from gymnasium.envs.mujoco import MujocoEnv 
import mujoco
from gymnasium import spaces
import os 
import collections

class S0100Env(MujocoEnv):
    metadata = {"render_modes": ["human", "rgb_array", "depth_array"]}
    
    def __init__(self, render_mode=None, max_episode_steps=200, success_distance=0.02, stage_distance=0.03, grab_distance=0.025, success_hold_steps=3, obs_type="blind", task_stage=1):
        xml_file = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "SO101", "block_stack.xml"))
        frame_skip = 20
        
        self.img_height = 84
        self.img_width = 84
        self.n_actuators = 6
        self.joint_obs_dim = 39
        
        self.obs_type = obs_type 
        self.task_stage = int(task_stage)

        self.observation_space = spaces.Dict({
            "joints": spaces.Box(low=-np.inf, high=np.inf, shape=(self.joint_obs_dim,), dtype=np.float32),
            "image": spaces.Box(low=0, high=255, shape=(self.img_height, self.img_width, 3), dtype=np.uint8)
        })
        
        # Action space: Outputs values between -1 and 1 and will be saled to actual robot limits
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(self.n_actuators,), dtype=np.float32)

        self.max_episode_steps = int(max_episode_steps)
        self.success_hold_steps = int(success_hold_steps) 
        self.success_tolerance = float(success_distance)
        self.stage_tolerance = float(max(stage_distance, self.success_tolerance))
        self.grab_tolerance = float(grab_distance)
        
        # Backward-compat alias used by older scripts.
        self.target_tolerance = self.success_tolerance
        self._episode_step = 0
        self._success_streak = 0
        self._prev_action = np.zeros(self.n_actuators, dtype=np.float32)
        
        super().__init__(model_path=xml_file, frame_skip=frame_skip, observation_space=self.observation_space, render_mode=render_mode)
        #self.mujoco_renderer = mujoco.Renderer(self.model, height=self.img_height, width=self.img_width)
        self.mujoco_renderer = None
        self.tip_site_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, "gripperframe")
        self.red_block_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, "block_red_site")
        self.blue_block_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, "block_blue_site")
        self.target_bottom_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, "target_bottom_site")
        self.target_top_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, "target_top_site")
        self.target_body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "target_stack")

        self.red_joint_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, "block_red_joint")
        self.blue_joint_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, "block_blue_joint")

        self.control_joint_ids = self.model.actuator_trnid[:self.n_actuators, 0].astype(np.int32)
        self.control_qpos_ids = np.asarray([self.model.jnt_qposadr[jid] for jid in self.control_joint_ids], dtype=np.int32)
        self.control_qvel_ids = np.asarray([self.model.jnt_dofadr[jid] for jid in self.control_joint_ids], dtype=np.int32)

    def _scale_action(self, action):
        action = np.asarray(action, dtype=np.float32)
        action = np.clip(action, -1.0, 1.0)
        ctrl_min = self.model.actuator_ctrlrange[:, 0]
        ctrl_max = self.model.actuator_ctrlrange[:, 1]
        action_scaled = ctrl_min + (action + 1.0) * 0.5 * (ctrl_max - ctrl_min)
        return np.clip(action_scaled, ctrl_min, ctrl_max)
    
    def get_distance(self):
        gripper_pos = self.data.site_xpos[self.tip_site_id]
        red_block_pos = self.data.site_xpos[self.red_block_id]
        blue_block_pos = self.data.site_xpos[self.blue_block_id]
        target_bottom_pos = self.data.site_xpos[self.target_bottom_id]
        target_top_pos = self.data.site_xpos[self.target_top_id]

        return {
        "dist_grab_red": float(np.linalg.norm(gripper_pos - red_block_pos)),
        "dist_place_red": float(np.linalg.norm(red_block_pos - target_bottom_pos)),
        "dist_grab_blue": float(np.linalg.norm(gripper_pos - blue_block_pos)),
        "dist_stack_blue": float(np.linalg.norm(blue_block_pos - target_top_pos)),
        }

    def compute_reward(self, gripper_pos, red_block_pos, blue_block_pos, target_bottom_pos, target_top_pos):
        # REACH: Move gripper to target
        if self.task_stage == 1:
            dist = np.linalg.norm(gripper_pos - target_bottom_pos)
            is_success = bool(dist < self.success_tolerance)
        
        # PUSH / PICK & PLACE: Red block to target (ignore blue)
        elif self.task_stage == 2 or self.task_stage == 3:
            dist = np.linalg.norm(red_block_pos - target_bottom_pos)
            is_success = bool(dist < self.success_tolerance)
        
        # STACK: Both blocks to their targets
        elif self.task_stage == 4:
            dist_red = np.linalg.norm(red_block_pos - target_bottom_pos)
            dist_blue = np.linalg.norm(blue_block_pos - target_top_pos)
            is_success = bool((dist_red < self.success_tolerance) and (dist_blue < self.success_tolerance))

        reward = 0.0 if is_success else -1.0
        return float(reward), is_success
        
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

        terminated = bool(self._success_streak >= self.success_hold_steps)
        truncated = bool(self._episode_step >= self.max_episode_steps and not terminated)
        
        distances = self.get_distance()

        if self.task_stage == 1:
            goal_distance = float(np.linalg.norm(gripper_pos - target_bottom_pos))
        elif self.task_stage in (2, 3):
            goal_distance = float(distances["dist_place_red"])
        else:
            goal_distance = float(max(distances["dist_place_red"], distances["dist_stack_blue"]))
        metrics = {
        "goal_distance": goal_distance,
        "dist_gripper_to_red": float(distances["dist_grab_red"]),
        "dist_red_to_bottom": float(distances["dist_place_red"]),
        "dist_gripper_to_blue": float(distances["dist_grab_blue"]),
        "dist_blue_to_top": float(distances["dist_stack_blue"]),
        }

        info = {
        "success": is_success,
        "success_streak": self._success_streak,
        "task_stage": int(self.task_stage),
        "task_name": {1: "reach", 2: "slide", 3: "pick_place", 4: "stack"}.get(self.task_stage, "unknown"),
        "metrics": metrics,
        "distances": distances,
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
        qpos[self.control_qpos_ids] += self.np_random.uniform(low=noise_low, high=noise_high, size=self.n_actuators)
        
        block_noise_range = 0.02

        target_x = float(self.np_random.uniform(0.10, 0.30))
        target_y = float(self.np_random.uniform(-0.20, 0.20))
        table_z = 0.02

        red_qpos_idx = self.model.jnt_qposadr[self.red_joint_id]
        blue_qpos_idx = self.model.jnt_qposadr[self.blue_joint_id]


        if self.task_stage == 1 or self.task_stage == 3:
            # REACH & PICK: Target is a random point in the air (5cm to 20cm up)
            target_z = float(self.np_random.uniform(0.05, 0.20))
        else:
            # PUSH & STACK: Target is flat on the table
            target_z = table_z

        # Move the static target body in the MuJoCo model
        self.model.body_pos[self.target_body_id] = np.array([target_x, target_y, target_z])

        # Hide objects that aren't part of the current stage
        if self.task_stage < 4:
            # Hide the blue block (Stages 1, 2, 3)
            qpos[blue_qpos_idx : blue_qpos_idx + 3] = np.array([10.0, 10.0, -2.0]) 
            qvel[blue_qpos_idx : blue_qpos_idx + 3] = 0.0
            
        if self.task_stage == 1:
            # Hide the red block (Stage 1 only)
            qpos[red_qpos_idx : red_qpos_idx + 3] = np.array([-10.0, -10.0, -2.0])
            qvel[red_qpos_idx : red_qpos_idx + 3] = 0.0
        else:
            # Stages 2, 3, 4: Spawn red block randomly on the table
            # (Prevents agent from memorizing the 0.15, 0.15 start pos)
            qpos[red_qpos_idx] = self.np_random.uniform(0.10, 0.30)     # X
            qpos[red_qpos_idx + 1] = self.np_random.uniform(-0.20, 0.20)  # Y
            qpos[red_qpos_idx + 2] = table_z

        # Set the state with the noisy positions
        self.set_state(qpos, qvel)
        mujoco.mj_forward(self.model, self.data)

        self._episode_step = 0
        self._success_streak = 0
        self._prev_action = np.zeros(self.n_actuators, dtype=np.float32)

        return self._get_obs()
    
    
    def _get_obs(self):
        joint_positions = self.data.qpos[self.control_qpos_ids].copy()
        joint_velocities = self.data.qvel[self.control_qvel_ids].copy()
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