import gym
import gymnasium
import numpy as np
from gym import spaces
from termcolor import cprint

import mani_skill.envs
from mani_skill.utils.structs import Actor, Link

from diffusion_policy_3d.gym_util.mjpc_wrapper import point_cloud_sampling


XY_CROP_MIN = -0.3
XY_CROP_MAX = 0.3


class ManiSkillEnv(gym.Env):
    metadata = {"render.modes": ["rgb_array"], "video.frames_per_second": 10}

    def __init__(self,
                 task_name,
                 max_episode_length=200,
                 image_size=128,
                 camera_name="base_camera",
                 obs_mode="pointcloud",
                 control_mode="pd_ee_delta_pose",
                 use_point_crop=False,
                 num_points=4096,
                 point_sample_method="fps",
                 render_mode="rgb_array",
                 num_envs=1,
                 use_pc_color=False,
                 filter_table_workspace=False):
        super().__init__()

        self.task_name = task_name
        self.max_episode_length = max_episode_length
        self.image_size = image_size
        self.camera_name = camera_name
        self.obs_mode = obs_mode
        self.control_mode = control_mode
        self.use_point_crop = use_point_crop
        self.num_points = num_points
        self.point_sample_method = point_sample_method
        self.render_mode = render_mode
        self.num_envs = num_envs
        self.use_pc_color = use_pc_color
        self.filter_table_workspace = filter_table_workspace
        self.cur_step = 0

        self.env = gymnasium.make(
            task_name,
            num_envs=num_envs,
            max_episode_steps=max_episode_length,
            obs_mode=obs_mode,
            control_mode=control_mode,
            render_mode=render_mode,
            sensor_configs={"width": image_size, "height": image_size},
        )

        self.action_space = self._convert_action_space(self.env.action_space)

        raw_obs, info = self.env.reset()
        self._last_raw_obs = self._squeeze_batch_dim(raw_obs)
        self._last_info = self._squeeze_batch_dim(info)
        self.obs_sensor_dim = self.get_robot_state().shape[0]
        point_cloud_dim = 6 if self.use_pc_color else 3

        self.observation_space = spaces.Dict({
            'image': spaces.Box(
                low=0,
                high=255,
                shape=(self.image_size, self.image_size, 3),
                dtype=np.uint8
            ),
            'agent_pos': spaces.Box(
                low=-np.inf,
                high=np.inf,
                shape=(self.obs_sensor_dim,),
                dtype=np.float32
            ),
            'point_cloud': spaces.Box(
                low=-np.inf,
                high=np.inf,
                shape=(self.num_points, point_cloud_dim),
                dtype=np.float32
            ),
        })

        cprint(f"[ManiSkillEnv] use_pc_color: {self.use_pc_color}", 'cyan')
        cprint(f"[ManiSkillEnv] point_sample_method: {self.point_sample_method}", 'cyan')

    def _convert_action_space(self, action_space):
        return spaces.Box(
            low=np.asarray(action_space.low, dtype=np.float32),
            high=np.asarray(action_space.high, dtype=np.float32),
            shape=action_space.shape,
            dtype=np.float32,
        )

    @staticmethod
    def _to_numpy(x):
        if hasattr(x, 'detach'):
            return x.detach().cpu().numpy()
        return np.asarray(x)

    @staticmethod
    def _to_scalar(x):
        if hasattr(x, 'detach'):
            x = x.detach().cpu().numpy()
        arr = np.asarray(x)
        if arr.size == 1:
            return arr.item()
        return arr

    def _squeeze_batch_dim(self, x):
        if self.num_envs != 1:
            return x
        if isinstance(x, dict):
            return {k: self._squeeze_batch_dim(v) for k, v in x.items()}
        if hasattr(x, 'shape') and len(x.shape) >= 1 and x.shape[0] == 1:
            if hasattr(x, 'detach'):
                return x.squeeze(0)
            return np.squeeze(x, axis=0)
        return x

    def _extract_success(self, info):
        info = self._squeeze_batch_dim(info)
        if not isinstance(info, dict):
            return False
        for key in ('success', 'is_success', 'episode_success', 'task_success', 'successes'):
            if key in info:
                try:
                    return bool(self._to_scalar(info[key]))
                except Exception:
                    return False
        return False

    def _build_full_point_cloud_and_mask(self):
        pc = self._last_raw_obs['pointcloud']
        xyzw = self._to_numpy(pc['xyzw']).astype(np.float32)
        rgb = self._to_numpy(pc['rgb']).astype(np.float32)

        xyz = xyzw[..., :3]
        valid_mask = xyzw[..., 3] > 0

        if rgb.max() <= 1.01:
            rgb = rgb * 255.0
        rgb = np.clip(rgb, 0, 255).astype(np.float32)

        point_cloud = np.concatenate([xyz, rgb], axis=-1)
        return point_cloud, valid_mask, pc

    def _get_segmentation_ids(self):
        base_env = getattr(self.env, 'unwrapped', self.env)
        seg_map = getattr(base_env, 'segmentation_id_map', {})

        robot_ids = set()
        ground_ids = set()
        table_ids = set()

        for obj_id, obj in seg_map.items():
            name = str(getattr(obj, 'name', '')).lower()
            obj_id = int(obj_id)
            if isinstance(obj, Link) and not isinstance(obj, Actor):
                robot_ids.add(obj_id)
            if isinstance(obj, Actor):
                if 'ground' in name:
                    ground_ids.add(obj_id)
                elif 'table-workspace' in name:
                    table_ids.add(obj_id)
        return robot_ids, ground_ids, table_ids

    def get_robot_state(self):
        obs = self._last_raw_obs
        tcp_pose = self._to_numpy(obs['extra']['tcp_pose']).astype(np.float32)
        qpos = self._to_numpy(obs['agent']['qpos']).astype(np.float32)
        gripper_width = qpos[..., -1:] + qpos[..., -2:-1]
        state = np.concatenate([tcp_pose.reshape(-1), gripper_width.reshape(-1)], axis=-1)
        return state.astype(np.float32)

    def get_rgb(self):
        rgb = self._to_numpy(self.env.render())
        rgb = self._squeeze_batch_dim(rgb)
        return rgb.astype(np.uint8)

    def get_point_cloud(self):
        point_cloud, valid_mask, pc_src = self._build_full_point_cloud_and_mask()
        seg = self._to_numpy(pc_src['segmentation'])
        if seg.shape[:1] != valid_mask.shape[:1]:
            seg = seg.reshape(valid_mask.shape[0], -1)
        if seg.ndim > 1:
            seg = seg[..., 0]
        seg = seg.reshape(-1)

        _, ground_ids, table_ids = self._get_segmentation_ids()
        remove_ids = set(ground_ids)
        if self.filter_table_workspace:
            remove_ids |= table_ids

        if len(remove_ids) > 0:
            keep_mask = (~np.isin(seg, np.array(list(remove_ids), dtype=seg.dtype))) & valid_mask
        else:
            keep_mask = valid_mask

        point_cloud = point_cloud[keep_mask]
        xy = point_cloud[:, :2]
        xy_keep_mask = (
            (xy[:, 0] >= XY_CROP_MIN)
            & (xy[:, 0] <= XY_CROP_MAX)
            & (xy[:, 1] >= XY_CROP_MIN)
            & (xy[:, 1] <= XY_CROP_MAX)
        )
        point_cloud = point_cloud[xy_keep_mask]
        point_cloud = point_cloud_sampling(point_cloud, self.num_points, self.point_sample_method)
        if not self.use_pc_color:
            point_cloud = point_cloud[..., :3]
        return point_cloud.astype(np.float32)

    def get_obs(self):
        return {
            'image': self.get_rgb(),
            'agent_pos': self.get_robot_state(),
            'point_cloud': self.get_point_cloud(),
        }

    def reset(self, **kwargs):
        self.cur_step = 0
        obs, info = self.env.reset(**kwargs)
        self._last_raw_obs = self._squeeze_batch_dim(obs)
        self._last_info = self._squeeze_batch_dim(info)
        return self.get_obs()

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        self._last_raw_obs = self._squeeze_batch_dim(obs)
        self._last_info = self._squeeze_batch_dim(info)

        reward = float(self._to_scalar(self._squeeze_batch_dim(reward)))
        terminated = bool(self._to_scalar(self._squeeze_batch_dim(terminated)))
        truncated = bool(self._to_scalar(self._squeeze_batch_dim(truncated)))

        self.cur_step += 1
        done = terminated or truncated or (self.cur_step >= self.max_episode_length)

        info = dict(self._last_info) if isinstance(self._last_info, dict) else {}
        info['success'] = self._extract_success(info)
        return self.get_obs(), reward, done, info

    def render(self, mode='rgb_array'):
        return self.get_rgb()

    def seed(self, seed=None):
        return seed

    def close(self):
        if self.env is not None:
            self.env.close()