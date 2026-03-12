#!/usr/bin/env python3
"""
Minimal script to run a short random-action ManiSkill evaluator and dump
input pointclouds as PLY. No CLI args; ensure you've activated the `dp3`
conda environment before running (e.g. `conda activate dp3`).
"""
import os
import sys
import pathlib

# Ensure package imports work: set cwd to the package folder
REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent
PKG_DIR = REPO_ROOT / "3D-Diffusion-Policy"
if not PKG_DIR.exists():
    PKG_DIR = REPO_ROOT
sys.path.insert(0, str(PKG_DIR))
os.chdir(str(PKG_DIR))

try:
    from diffusion_policy_3d.env_runner.maniskill_runner import ManiSkillRunner
except Exception as e:
    print("Failed to import ManiSkillRunner. Make sure you're running inside the project and dependencies are installed.")
    raise


def main():
    output_dir = pathlib.Path.cwd() / 'debug_eval_random'
    output_dir.mkdir(parents=True, exist_ok=True)

    runner = ManiSkillRunner(
        output_dir=str(output_dir),
        eval_episodes=1,
        max_steps=16,
        n_obs_steps=8,
        n_action_steps=8,
        n_envs=1,
        task_name='PickCube-v1',
        image_size=128,
        camera_name='base_camera',
        obs_mode='pointcloud',
        control_mode='pd_ee_delta_pose',
        num_points=1024,
        point_sample_method='fps',
        render_mode='rgb_array',
        use_pc_color=True,
        filter_table_workspace=False,
        save_pointcloud_ply=True,
        pointcloud_ply_dir='input_pointcloud_ply',
        pointcloud_ply_max_steps=3,
        use_random_action=True,
        print_robot_states=True,
        robot_state_print_max_steps=3,
    )

    log_data = runner.run(policy=None)
    print('RUNNER_LOG', log_data)
    print('PLY_DIR', output_dir / 'input_pointcloud_ply')
    try:
        runner.env.close()
    except Exception:
        pass


if __name__ == '__main__':
    main()
