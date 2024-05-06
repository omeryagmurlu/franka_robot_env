import copy
import logging
import os
import time
from typing import Any, Dict, Tuple, Union
import cv2

import gym
import numpy as np
import torch
from PIL import Image
import matplotlib.pyplot as plt

from mdt.datasets.utils.episode_utils import process_depth, process_rgb, process_state

logger = logging.getLogger(__name__)

class RealRobotEnv(gym.Env):
    def __init__(self, 
                device,
                observation_space_keys,
                transforms,
                proprio_state,
                robot_ip,
                top_cam_mxid,
                second_cam_mxid,
                use_delta_robot = False,
                starting_pos_noise = 0.0,
                 **kwargs):
        # env = get_env(
        #     dataset_loader.abs_datasets_dir, show_gui=show_gui, obs_space=dataset_loader.observation_space, **kwargs
        # )
        super(RealRobotEnv, self).__init__()
        
        # self.observation_space_keys = observation_space_keys
        self.observation_space_keys = dict(
            depth_obs = [],
            # rgb_obs = ['agentview_rgb', 'eye_in_hand_rgb', 'gen_static', 'gen_gripper'], # I'm really not sure what's going on with these but whatever
            rgb_obs = ['rgb_static', 'rgb_gripper'],
            state_obs = ['joint_pos'],
            actions = ['rel_actions'] # ?????
        )
        self.transforms = transforms
        # TODO change hgere
        # self.transforms['val']['rgb_gripper'] = self.transforms['val']['rgb_static']
        self.proprio_state = proprio_state
        self.device = device
        self.starting_pos_noise = starting_pos_noise
        self.use_delta_robot = use_delta_robot
        self.relative_actions = "rel_actions" in self.observation_space_keys["actions"]

        self.prev_time = time.time()
        self.hz = 6

        self.velocity_limits = True

        self.des_joint_state_hist = []
        self.joint_state_hist = []
        self.timesteps = []

        # default_reset_qpos = [ 0.0000,  0.0000,  0.0000, -1.7359,  0.0000,  1.7985,  0.7854]
        # default_reset_qpos = [ 0.24326089, -0.049953, 0.64464317, -1.57563733, -0.30243619, 1.8057968, -1.42613693]
        default_reset_qpos = [ 0.2719, -0.5165,  0.2650, -1.6160, -0.0920,  1.6146, -1.9760]
        self.default_reset_gripper = 1
        self.default_reset_qpos = default_reset_qpos

        self.robot = FrankaArm(
            name='franka', 
            ip_address=robot_ip, 
            default_reset_pose=default_reset_qpos, 
            hz=6, # self.hz,
        )
        self.hand = FrankaHand(name='frankahand', ip_address=robot_ip)
        # kp_new = 1.5* torch.Tensor(self.robot.metadata.default_Kq)
        # kd_new = 1.5* torch.Tensor(self.robot.metadata.default_Kqd)
        # self.robot.apply_commands(kp=kp_new, kd=kd_new)
        self.top_cam = DepthAI(name="topcam", device_MxId=top_cam_mxid)
        self.second_cam = DepthAI(name="secondcam", device_MxId=second_cam_mxid)

        assert self.top_cam.connect(), "Connection to top_cam failed."
        assert self.second_cam.connect(), "Connection to second_cam failed."
        assert self.robot.connect(policy=None), "Connection to robot failed."
        assert self.hand.connect(policy=None), "Connection to hand failed."

    def transform_observation(self, obs: Dict[str, Any]) -> Dict[str, Union[torch.Tensor, Dict[str, torch.Tensor]]]:
        # proprioception_dims: none in real_kitchen.yaml, but robot_obs are still in the batches for training? ask tomorrow. Forward ignores them, so its okay
        state_obs = process_state(obs, self.observation_space_keys, self.transforms['val'], self.proprio_state)
        rgb_obs = process_rgb(obs["rgb_obs"], self.observation_space_keys, self.transforms['val'])
        depth_obs = process_depth(obs["depth_obs"], self.observation_space_keys, self.transforms['val'])

        state_obs["robot_obs"] = state_obs["robot_obs"].to(self.device).unsqueeze(0)
        rgb_obs.update({"rgb_obs": {k: v.to(self.device).unsqueeze(0) for k, v in rgb_obs["rgb_obs"].items()}})
        depth_obs.update({"depth_obs": {k: v.to(self.device).unsqueeze(0) for k, v in depth_obs["depth_obs"].items()}})

        obs_dict: Dict = {
            **rgb_obs,
            **state_obs,
            **depth_obs,
            "robot_obs_raw": torch.from_numpy(obs["joint_pos"]).to(self.device),
            "rgb_obs_raw": obs["rgb_obs"],
        }
        return obs_dict
    
    def step(
        self, action_tensor: torch.Tensor
    ) -> Tuple[Dict[str, Union[torch.Tensor, Dict[str, torch.Tensor]]], int, bool, Dict]:
        if self.relative_actions:
            action = action_tensor.squeeze().cpu().detach().numpy()
            # assert len(action) == 7
        else:
            if action_tensor.shape[-1] == 7:
                slice_ids = [3, 6]
            elif action_tensor.shape[-1] == 8:
                slice_ids = [3, 7]
            else:
                logger.error("actions are required to have length 8 (for euler angles) or 9 (for quaternions)")
                raise NotImplementedError
            action = np.split(action_tensor.squeeze().cpu().detach().numpy(), slice_ids)
        action[-1] = 1 if action[-1] > 0 else -1

        assert action.shape == (8,)
        

        # TODO: FIXME sync hardware time
        curr_time = time.time()
        elapsed = max(curr_time - self.prev_time, 0)
        self.prev_time = curr_time

        duration = 1 / self.hz
        remaining = max(duration - elapsed, 0)
        if remaining == 0:
            print("Step took %0.4fs, step duration: %0.4fs"%(elapsed, duration))
        else:
            time.sleep(remaining)
        # print(action)
            
        robot_action = action[:7]
        if self.use_delta_robot:
            robot_action = self.robot.get_sensors()['joint_pos'].numpy() + robot_action
            
        if self.velocity_limits:
            q_des_unclipped = robot_action
            q_desired, clip_ttg_multiplier = self.robot.clip_velocity_limits(q_des_unclipped)
            if not np.allclose(q_des_unclipped, q_desired):
                print("WARNING: Clipping velocity limits, desired norm: %s, clipped norm: %s"%(np.linalg.norm(q_des_unclipped), np.linalg.norm(q_desired)))
            # clip_ttg_multiplier = np.ceil(clip_ttg_multiplier)
        else:
            q_desired = robot_action
            clip_ttg_multiplier = 1

        self.des_joint_state_hist.append(q_desired)
        self.joint_state_hist.append(self.robot.get_sensors()['joint_pos'].numpy())
        self.timesteps.append(time.time())

        self.robot.follow_trajectory(
            joint_positions=robot_action,
            time_to_go=1 * clip_ttg_multiplier,
            converge_pos_err=False,
            )

        # self.robot.follow_trajectory(
        #     joint_positions=q_desired,
        #     time_to_go=1,
        #     converge_pos_err=False,
        #     )
        # print(action[7])
        self.hand.apply_commands(width=action[7])

        obs = self.get_obs()
        return obs, 0, False, {}
    
    def step_2(
        self, action_tensor: torch.Tensor
    ) -> Tuple[Dict[str, Union[torch.Tensor, Dict[str, torch.Tensor]]], int, bool, Dict]:
        if self.relative_actions:
            action = action_tensor.squeeze().cpu().detach().numpy()
            # assert len(action) == 7
        else:
            if action_tensor.shape[-1] == 7:
                slice_ids = [3, 6]
            elif action_tensor.shape[-1] == 8:
                slice_ids = [3, 7]
            else:
                logger.error("actions are required to have length 8 (for euler angles) or 9 (for quaternions)")
                raise NotImplementedError
            action = np.split(action_tensor.squeeze().cpu().detach().numpy(), slice_ids)
        action[-1] = 1 if action[-1] > 0 else -1

        assert action.shape == (8,)
        

        # TODO: FIXME sync hardware time
        curr_time = time.time()
        elapsed = max(curr_time - self.prev_time, 0)
        self.prev_time = curr_time

        duration = 1 / self.hz
        remaining = max(duration - elapsed, 0)
        if remaining == 0:
            print("Step took %0.4fs, step duration: %0.4fs"%(elapsed, duration))
        else:
            time.sleep(remaining)
        # print(action)
            
        robot_action = action[:7]
        if self.use_delta_robot:
            robot_action = self.robot.get_sensors()['joint_pos'].numpy() + robot_action
            
        if self.velocity_limits:
            q_des_unclipped = robot_action
            q_desired = self.robot.clip_velocity_limits(q_des_unclipped)
            if not np.allclose(q_des_unclipped, q_desired):
                print("WARNING: Clipping velocity limits, desired: %s, clipped: %s, abs diff: %s"%(q_des_unclipped, q_desired, np.abs(q_des_unclipped - q_desired)))
        else:
            q_desired = robot_action

        self.des_joint_state_hist.append(q_desired)
        self.joint_state_hist.append(self.robot.get_sensors()['joint_pos'].numpy())
        self.timesteps.append(time.time())

        self.robot.apply_commands(q_desired=q_desired)
        self.hand.apply_commands(width=action[7])

        obs = self.get_obs()
        return obs, 0, False, {}

    def reset(
        self,
        reset_info: Dict[str, Any] = None,
        batch_idx: int = 0,
        seq_idx: int = 0,
        # scene_obs: Any = None,
        robot_obs: Any = None,
    ) -> Dict[str, Union[torch.Tensor, Dict[str, torch.Tensor]]]:
        if reset_info is not None:
            raise NotImplementedError
            # self.env.reset(
            #     robot_obs=reset_info["robot_obs"][batch_idx, seq_idx],
            #     scene_obs=reset_info["scene_obs"][batch_idx, seq_idx],
            # )
        elif robot_obs is not None:
            assert robot_obs.shape == (8,), "robot_obs needs to have a shape of (8,)"
            self.robot.reset(reset_pos=robot_obs[:7])
            self.hand.reset(width=robot_obs[7])
        else:
            rand_jitter = np.random.uniform(-self.starting_pos_noise, self.starting_pos_noise, size=np.shape(np.array(self.default_reset_qpos)))
            print('rand_jitter', rand_jitter)
            reset_des_pos = (np.array(self.default_reset_qpos) + rand_jitter).tolist()
            self.robot.reset(reset_pos=reset_des_pos)
            self.hand.reset(width=self.default_reset_gripper)

        return self.get_obs()

    def get_info(self):
        return {}

    def get_obs(self):
        joint_pos = self.robot.get_sensors()['joint_pos'].numpy()
        top_rgb = self.top_cam.get_sensors()['rgb']
        second_rgb = self.second_cam.get_sensors()['rgb']

        top_rgb = resize_and_crop(top_rgb, "top_center_new_lab", des_height=250, des_width=250)
        second_rgb = resize_and_crop(second_rgb, "front_center_new_lab", des_height=250, des_width=250)

        obs = dict(
            joint_pos = joint_pos,
            rgb_obs = dict(
                rgb_static = top_rgb,
                rgb_gripper = second_rgb
            ),
            depth_obs = []
        )

        return self.transform_observation(obs)
