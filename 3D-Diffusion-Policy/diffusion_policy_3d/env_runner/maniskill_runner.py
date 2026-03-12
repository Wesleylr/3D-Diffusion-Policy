import pathlib

import numpy as np
import torch
import tqdm
import wandb
from termcolor import cprint

import diffusion_policy_3d.common.logger_util as logger_util
from diffusion_policy_3d.common.pytorch_util import dict_apply
from diffusion_policy_3d.env.maniskill import ManiSkillEnv
from diffusion_policy_3d.env_runner.base_runner import BaseRunner
from diffusion_policy_3d.gym_util.multistep_wrapper import MultiStepWrapper
from diffusion_policy_3d.gym_util.video_recording_wrapper import SimpleVideoRecordingWrapper
from diffusion_policy_3d.policy.base_policy import BasePolicy


class ManiSkillRunner(BaseRunner):
    def __init__(self,
                 output_dir,
                 eval_episodes=20,
                 max_steps=200,
                 n_obs_steps=8,
                 n_action_steps=8,
                 fps=10,
                 crf=22,
                 tqdm_interval_sec=5.0,
                 n_envs=1,
                 task_name=None,
                 image_size=128,
                 camera_name='base_camera',
                 obs_mode='pointcloud',
                 control_mode='pd_ee_delta_pose',
                 use_point_crop=False,
                 num_points=1024,
                 point_sample_method='fps',
                 render_mode='rgb_array',
                 use_pc_color=False,
                 filter_table_workspace=False,
                 save_pointcloud_ply=False,
                 pointcloud_ply_dir='eval_pointcloud_ply',
                 pointcloud_ply_max_steps=None,
                 use_random_action=False,
                 print_robot_states=False,
                 robot_state_print_max_steps=None):
        super().__init__(output_dir)
        self.task_name = task_name
        self.save_pointcloud_ply = save_pointcloud_ply
        self.pointcloud_ply_dir = pathlib.Path(output_dir) / pointcloud_ply_dir
        self.pointcloud_ply_max_steps = pointcloud_ply_max_steps
        self.use_random_action = use_random_action
        self.print_robot_states = print_robot_states
        self.robot_state_print_max_steps = robot_state_print_max_steps

        def env_fn(task_name):
            return MultiStepWrapper(
                SimpleVideoRecordingWrapper(
                    ManiSkillEnv(
                        task_name=task_name,
                        max_episode_length=max_steps,
                        image_size=image_size,
                        camera_name=camera_name,
                        obs_mode=obs_mode,
                        control_mode=control_mode,
                        use_point_crop=use_point_crop,
                        num_points=num_points,
                        point_sample_method=point_sample_method,
                        render_mode=render_mode,
                        num_envs=n_envs,
                        use_pc_color=use_pc_color,
                        filter_table_workspace=filter_table_workspace,
                    )
                ),
                n_obs_steps=n_obs_steps,
                n_action_steps=n_action_steps,
                max_episode_steps=max_steps,
                reward_agg_method='sum',
            )

        self.eval_episodes = eval_episodes
        self.env = env_fn(self.task_name)

        self.fps = fps
        self.crf = crf
        self.n_obs_steps = n_obs_steps
        self.n_action_steps = n_action_steps
        self.max_steps = max_steps
        self.tqdm_interval_sec = tqdm_interval_sec

        self.logger_util_test = logger_util.LargestKRecorder(K=3)
        self.logger_util_test10 = logger_util.LargestKRecorder(K=5)

    @staticmethod
    def _save_point_cloud_ply(point_cloud: np.ndarray, save_path: pathlib.Path):
        point_cloud = np.asarray(point_cloud, dtype=np.float32)
        xyz = point_cloud[:, :3]

        with open(save_path, 'w', encoding='utf-8') as file:
            file.write('ply\n')
            file.write('format ascii 1.0\n')
            file.write(f'element vertex {xyz.shape[0]}\n')
            file.write('property float x\n')
            file.write('property float y\n')
            file.write('property float z\n')

            has_color = point_cloud.shape[1] >= 6
            if has_color:
                file.write('property uchar red\n')
                file.write('property uchar green\n')
                file.write('property uchar blue\n')

            file.write('end_header\n')

            if has_color:
                rgb = np.clip(point_cloud[:, 3:6], 0, 255).astype(np.uint8)
                for xyz_row, rgb_row in zip(xyz, rgb):
                    x, y, z = xyz_row
                    r, g, b = rgb_row
                    file.write(f'{x} {y} {z} {int(r)} {int(g)} {int(b)}\n')
            else:
                for xyz_row in xyz:
                    x, y, z = xyz_row
                    file.write(f'{x} {y} {z}\n')

    def _dump_obs_point_clouds(self, obs, episode_idx: int, step_idx: int):
        if not self.save_pointcloud_ply:
            return
        if self.pointcloud_ply_max_steps is not None and step_idx >= self.pointcloud_ply_max_steps:
            return

        point_cloud_obs = np.asarray(obs['point_cloud'])
        episode_dir = self.pointcloud_ply_dir / f'episode_{episode_idx:03d}'
        episode_dir.mkdir(parents=True, exist_ok=True)

        for obs_idx, point_cloud in enumerate(point_cloud_obs):
            save_path = episode_dir / f'step_{step_idx:03d}_obs_{obs_idx:02d}.ply'
            self._save_point_cloud_ply(point_cloud, save_path)

    def _print_robot_state_input(self, obs, episode_idx: int, step_idx: int):
        if not self.print_robot_states:
            return
        if self.robot_state_print_max_steps is not None and step_idx >= self.robot_state_print_max_steps:
            return

        agent_pos = np.asarray(obs['agent_pos'], dtype=np.float32)
        cprint(
            f"[EvalInput] episode={episode_idx} step={step_idx} agent_pos shape={agent_pos.shape}",
            'yellow'
        )
        print(agent_pos)

    def run(self, policy: BasePolicy = None, save_video=False):
        device = None if policy is None else policy.device

        all_traj_rewards = []
        all_success_rates = []
        env = self.env

        if self.save_pointcloud_ply:
            self.pointcloud_ply_dir.mkdir(parents=True, exist_ok=True)
            cprint(f"Saving eval point clouds to {self.pointcloud_ply_dir}", 'cyan')
        if self.use_random_action:
            cprint('Using random actions during evaluation.', 'cyan')

        for episode_idx in tqdm.tqdm(
                range(self.eval_episodes),
                desc=f"Eval in ManiSkill {self.task_name} Pointcloud Env",
                leave=False,
                mininterval=self.tqdm_interval_sec):
            obs = env.reset()
            if policy is not None:
                policy.reset()
            step_idx = 0
            self._dump_obs_point_clouds(obs, episode_idx=episode_idx, step_idx=step_idx)
            self._print_robot_state_input(obs, episode_idx=episode_idx, step_idx=step_idx)

            done = False
            traj_reward = 0
            is_success = False
            while not done:
                if self.use_random_action:
                    action = env.action_space.sample()
                else:
                    np_obs_dict = dict(obs)
                    obs_dict = dict_apply(
                        np_obs_dict,
                        lambda x: torch.from_numpy(x).to(device=device)
                    )

                    with torch.no_grad():
                        obs_dict_input = {
                            'point_cloud': obs_dict['point_cloud'].unsqueeze(0),
                            'agent_pos': obs_dict['agent_pos'].unsqueeze(0),
                        }
                        action_dict = policy.predict_action(obs_dict_input)

                    np_action_dict = dict_apply(
                        action_dict,
                        lambda x: x.detach().to('cpu').numpy()
                    )
                    action = np_action_dict['action'].squeeze(0)

                obs, reward, done, info = env.step(action)
                step_idx += 1
                self._dump_obs_point_clouds(obs, episode_idx=episode_idx, step_idx=step_idx)
                self._print_robot_state_input(obs, episode_idx=episode_idx, step_idx=step_idx)
                traj_reward += reward
                done = np.all(done)
                if 'success' in info:
                    is_success = is_success or bool(np.max(info['success']))

            all_success_rates.append(is_success)
            all_traj_rewards.append(traj_reward)

        log_data = dict()
        log_data['mean_traj_rewards'] = np.mean(all_traj_rewards)
        log_data['mean_success_rates'] = np.mean(all_success_rates)
        log_data['test_mean_score'] = np.mean(all_success_rates)

        cprint(f"test_mean_score: {np.mean(all_success_rates)}", 'green')

        self.logger_util_test.record(np.mean(all_success_rates))
        self.logger_util_test10.record(np.mean(all_success_rates))
        log_data['SR_test_L3'] = self.logger_util_test.average_of_largest_K()
        log_data['SR_test_L5'] = self.logger_util_test10.average_of_largest_K()

        videos = env.env.get_video()
        if len(videos.shape) == 5:
            videos = videos[:, 0]

        if save_video:
            log_data['sim_video_eval'] = wandb.Video(videos, fps=self.fps, format='mp4')

        _ = env.reset()
        videos = None

        return log_data