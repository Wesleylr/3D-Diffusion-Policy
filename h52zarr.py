
from __future__ import annotations
import argparse
import importlib.util
import os
import pathlib
import sys
import numpy as np
import h5py
import zarr
import tqdm
from termcolor import cprint, colored

PROJECT_ROOT = pathlib.Path(__file__).resolve().parent / "3D-Diffusion-Policy"
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def _load_point_cloud_sampling():
    wrapper_path = PROJECT_ROOT / "diffusion_policy_3d" / "gym_util" / "mjpc_wrapper.py"
    spec = importlib.util.spec_from_file_location("dp3_mjpc_wrapper", wrapper_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.point_cloud_sampling


point_cloud_sampling = _load_point_cloud_sampling()

# 兼容原有代码的常量配置
DEFAULT_COMPRESSOR = zarr.Blosc(cname="zstd", clevel=3, shuffle=1)
DEFAULT_IMAGE_SHAPE = (512, 512, 3)  # 与你生成代码默认image_size一致
DEFAULT_POINT_CLOUD_NPOINTS = 1024
DEFAULT_ACTION_DIM = 7  # 匹配你H5中actions的4维动作
DEFAULT_GROUND_SEGMENTATION_IDS = {17}
DEFAULT_XY_CROP_MIN = -0.3
DEFAULT_XY_CROP_MAX = 0.3


def crop_point_cloud_xy(
    point_cloud: np.ndarray,
    x_min: float = DEFAULT_XY_CROP_MIN,
    x_max: float = DEFAULT_XY_CROP_MAX,
    y_min: float = DEFAULT_XY_CROP_MIN,
    y_max: float = DEFAULT_XY_CROP_MAX,
) -> np.ndarray:
    xy = point_cloud[:, :2]
    keep_mask = (
        (xy[:, 0] >= x_min)
        & (xy[:, 0] <= x_max)
        & (xy[:, 1] >= y_min)
        & (xy[:, 1] <= y_max)
    )
    return point_cloud[keep_mask]

def filter_point_cloud_by_segmentation(
    point_cloud: np.ndarray,
    seg_ids: np.ndarray,
    valid_mask: np.ndarray,
    remove_seg_ids: set[int],
    num_points: int = DEFAULT_POINT_CLOUD_NPOINTS,
    sample_method: str = "fps",
) -> np.ndarray:
    """
    使用与 ManiSkill wrapper 一致的方式过滤并降采样单帧点云。
    Args:
        point_cloud: 原始点云，shape=(N, 6)，字段为xyzrgb (float32)
        seg_ids: 每个点对应的 segmentation id，shape=(N,)
        valid_mask: 每个点是否有效，shape=(N,)
        remove_seg_ids: 需要剔除的 segmentation id 集合
        num_points: 最终采样的点云点数，默认4096
        sample_method: 点云采样方法，默认fps
    Returns:
        sampled_pc: 过滤并采样后的点云，shape=(num_points, 6)，float32
    """
    seg_ids = seg_ids.reshape(-1)
    valid_mask = valid_mask.reshape(-1).astype(bool)

    keep_mask = valid_mask
    if len(remove_seg_ids) > 0:
        remove_ids_np = np.array(list(remove_seg_ids), dtype=seg_ids.dtype)
        keep_mask = keep_mask & (~np.isin(seg_ids, remove_ids_np))

    filtered_pc = point_cloud[keep_mask]
    filtered_pc = crop_point_cloud_xy(filtered_pc)
    sampled_pc = point_cloud_sampling(filtered_pc, num_points, sample_method)
    return sampled_pc.astype(np.float32)


def extract_traj_data(traj_group, camera_name: str = "base_camera"):
    """
    从单条轨迹Group中提取核心数据
    :param traj_group: h5py的traj_x Group对象
    :param camera_name: 相机名称（匹配你H5中的sensor_param/base_camera）
    :return: 单轨迹的各类数据列表
    """
    obs_group = traj_group["obs"]
    actions = traj_group["actions"][:]  # (T, 4) T为动作步数

    # 1. 提取点云数据 (观测步数=T+1 → 取前T步匹配动作步数)
    pcd_group = obs_group["pointcloud"]
    pcd_xyzw = pcd_group["xyzw"][:-1]
    pcd_xyz = pcd_xyzw[:, :, :3]  # (T, N, 3) 剔除最后一帧观测，匹配动作步数
    pcd_rgb = pcd_group["rgb"][:-1]    # (T, N, 3)
    valid_masks = pcd_xyzw[:, :, 3] > 0
    # 合并点云坐标+颜色 → (T, N, 6)，与原有代码point_cloud格式对齐
    # 合并点云坐标+颜色 → (T, N, 6)，与原有代码point_cloud格式对齐
    point_clouds = np.concatenate([pcd_xyz, pcd_rgb.astype(np.float32)], axis=-1)
    segs = pcd_group["segmentation"][:-1]
    if segs.ndim > 2:
        segs = segs[..., 0]

    def _filter_seq(
        pc_seq: np.ndarray,
        seg_seq: np.ndarray,
        valid_mask_seq: np.ndarray,
        remove_ids: set[int],
    ) -> np.ndarray:
        """Apply wrapper-aligned point cloud filtering for each timestep and stack results.
        pc_seq: (T, N, 6)
        seg_seq: (T, N)
        valid_mask_seq: (T, N)
        returns: (T, num_points, 6)
        """
        sampled = []
        for pc_frame, seg_frame, valid_mask_frame in zip(pc_seq, seg_seq, valid_mask_seq):
            sampled_frame = filter_point_cloud_by_segmentation(
                point_cloud=pc_frame,
                seg_ids=seg_frame,
                valid_mask=valid_mask_frame,
                remove_seg_ids=remove_ids,
                num_points=DEFAULT_POINT_CLOUD_NPOINTS,
            )
            sampled.append(sampled_frame)
        return np.stack(sampled, axis=0)

    point_cloud = _filter_seq(
        point_clouds,
        segs,
        valid_masks,
        remove_ids=DEFAULT_GROUND_SEGMENTATION_IDS,
    )
    
    extra = obs_group["extra"]
    tcp_pose = extra["tcp_pose"][:-1]  # (T, 7) 取前T步匹配动作步数

    # 2. 提取机器人状态 (qpos+qvel → 合并为机器人状态，取前T步)
    agent_group = obs_group["agent"]
    qpos = agent_group["qpos"][:-1]  # (T, 9)
    gripper_width = qpos[..., -1:] + qpos[..., -2:-1]
    robot_state = np.concatenate([tcp_pose, gripper_width], axis=-1)  # (T, 18) 可根据需求调整维度

    return {
        "point_cloud": point_cloud,
        "robot_state": robot_state,
        "action": actions,
    }



def h52zarr_single(h5_path: str, zarr_save_dir: str, camera_name: str = "base_camera", quiet: bool = False):
    """
    单H5文件转Zarr
    :param h5_path: H5文件路径
    :param zarr_save_dir: Zarr保存根目录
    :param camera_name: 相机名称
    :param quiet: 是否静默模式
    """
    h5_path = pathlib.Path(h5_path)
    task_id = h5_path.parent.name  # 从H5父目录提取任务名（如PickCube-v1）
    zarr_filename = f"{task_id}_{camera_name}.zarr"
    zarr_path = pathlib.Path(zarr_save_dir) / zarr_filename
    zarr_path.mkdir(parents=True, exist_ok=True)

    # 初始化数据容器
    all_data = {
        "point_cloud": [],  # use singular to match ManiSkillDataset
        "robot_state": [],
        "action": [],
        "episode_ends": []  # 记录每个轨迹的结束步索引
    }
    total_step = 0

    # 读取H5文件
    with h5py.File(h5_path, "r") as f:
        traj_keys = sorted([k for k in f.keys() if k.startswith("traj_")])
        if not traj_keys:
            cprint(f"❌ {h5_path.name} 中未找到轨迹（traj_*）", "red")
            return
        cprint(f"📂 开始转换 {h5_path.name}，共{len(traj_keys)}条轨迹", "blue")

        # 遍历每条轨迹
        traj_iter = tqdm.tqdm(traj_keys) if not quiet else traj_keys
        for traj_key in traj_iter:
            traj_group = f[traj_key]
            # 提取单轨迹数据
            traj_data = extract_traj_data(traj_group, camera_name)
            # 打印每条轨迹的尺寸信息
            steps = traj_data["action"].shape[0]
            pc_shape = traj_data["point_cloud"].shape
            rs_shape = traj_data["robot_state"].shape
            act_shape = traj_data["action"].shape
            cprint(
                f"  {traj_key}: steps={steps}, pc={pc_shape},"
                f" state={rs_shape}, action={act_shape}",
                "green",
            )
            # 数据校验
            # 追加到全局容器
            for k in all_data.keys():
                if k == "episode_ends":
                    continue
                all_data[k].append(traj_data[k])
            # 更新累计步数和轨迹结束索引
            total_step += len(traj_data["action"])
            all_data["episode_ends"].append(total_step)

    # 合并所有轨迹数据为numpy数组
    cprint(f"\n📥 合并所有轨迹数据，总步数：{total_step}", "blue")
    for k in all_data.keys():
        if k == "episode_ends":
            all_data[k] = np.array(all_data[k], dtype=np.int64)
        elif k == "texts":
            all_data[k] = np.array(sum(all_data[k], []), dtype=object)  # 展平列表
        else:
            all_data[k] = np.concatenate(all_data[k], axis=0)


    # 创建Zarr文件（与你原有代码结构完全一致：data/ + meta/）
    cprint(f"📝 写入Zarr文件：{zarr_path}", "blue")
    zarr_root = zarr.group(zarr_path, overwrite=True)
    zarr_data = zarr_root.create_group("data", overwrite=True)
    zarr_meta = zarr_root.create_group("meta", overwrite=True)

    # 定义分块大小（与你原有代码一致，按100步分块），根据实际数据shape自动推断最后两维
    chunk_config = {
        "images": (100, *DEFAULT_IMAGE_SHAPE),
        "point_cloud": (100, all_data["point_cloud"].shape[1], all_data["point_cloud"].shape[2]),
        "robot_state": (100, all_data["robot_state"].shape[1]),
        "action": (100, all_data["action"].shape[1]),
        "rewards": (100,),
    }

    # 写入data组数据
    for k in ["point_cloud", "robot_state", "action"]:
        zarr_data.create_dataset(
            name=k,
            data=all_data[k],
            chunks=chunk_config[k],
            dtype=all_data[k].dtype,
            compressor=DEFAULT_COMPRESSOR,
        )
    # 写入文本数据（用MsgPack编码，与你原有代码一致）
    # 写入meta组的episode_ends
    zarr_meta.create_dataset(
        name="episode_ends",
        data=all_data["episode_ends"],
        dtype=np.int64,
        compressor=DEFAULT_COMPRESSOR
    )

    # 为每个episode保存第一步点云为PLY文件
    def _save_point_cloud_ply(point_cloud_xyzrgb: np.ndarray, save_path: pathlib.Path):
        """Write a simple ASCII PLY for xyzrgb point cloud."""
        pc = np.asarray(point_cloud_xyzrgb, dtype=np.float32)
        xyz = pc[:, :3]
        rgb = np.clip(pc[:, 3:6], 0, 255).astype(np.uint8)
        with open(save_path, "w", encoding="utf-8") as f:
            f.write("ply\n")
            f.write("format ascii 1.0\n")
            f.write(f"element vertex {xyz.shape[0]}\n")
            f.write("property float x\n")
            f.write("property float y\n")
            f.write("property float z\n")
            f.write("property uchar red\n")
            f.write("property uchar green\n")
            f.write("property uchar blue\n")
            f.write("end_header\n")
            for i in range(xyz.shape[0]):
                x, y, z = xyz[i]
                r, g, b = rgb[i]
                f.write(f"{x} {y} {z} {int(r)} {int(g)} {int(b)}\n")

    # 计算每个episode起始全局步数并存储点云
    ep_ends = all_data["episode_ends"]
    ep_starts = [0] + ep_ends[:-1].tolist()
    ply_dir = zarr_path / "first_step_pcs"
    ply_dir.mkdir(exist_ok=True)
    for ep_idx, start in enumerate(ep_starts):
        pc_with = all_data["point_cloud"][start]  # (N,6) xyzrgb
        ply_path1 = ply_dir / f"episode_{ep_idx:03d}_first_with_robot.ply"
        _save_point_cloud_ply(pc_with, ply_path1)
    cprint(f"✅ 转换完成！Zarr文件结构：", "green")
    print(zarr_root.tree())
    # cprint(f"📁 第一帧点云PLY已保存到：{ply_dir}", "green")
    return zarr_path


def main(args):
    # 初始化保存目录
    pathlib.Path(args.zarr_save_dir).mkdir(parents=True, exist_ok=True)
    cprint(f"📌 目标Zarr保存目录：{args.zarr_save_dir}", "yellow")

    # 单文件/批量转换
    if os.path.isfile(args.input_path):
        h52zarr_single(args.input_path, args.zarr_save_dir, args.camera_name, args.quiet)
    else:
        cprint(f"❌ 输入路径无效：{args.input_path}", "red")
        return

    cprint(f"\n🎉 所有转换任务完成！", "green")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="ManiSkill H5转Zarr脚本（适配多轨迹H5结构）")
    parser.add_argument(
        "--input-path",
        type=str,
        required=True,
        help="H5文件路径 或 存放H5文件的目录（支持批量转换）"
    )
    parser.add_argument(
        "--zarr-save-dir",
        type=str,
        default=str(pathlib.Path(__file__).resolve().parent / "data_new" / "maniskill_zarr"),
        help="Zarr文件保存根目录（默认与你原有代码data_new同级）"
    )
    parser.add_argument(
        "--camera-name",
        type=str,
        default="base_camera",
        help="相机名称（匹配你H5中的sensor_param/base_camera）"
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="静默模式（关闭进度条和详细打印）"
    )
    args = parser.parse_args()
    main(args)