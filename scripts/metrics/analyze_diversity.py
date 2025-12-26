import h5py
import numpy as np
import matplotlib.pyplot as plt
from sklearn.neighbors import NearestNeighbors
from scipy.spatial.distance import pdist


def analyze_diversity_final(hdf5_path, index=1):
    print(f"{'=' * 40}\n📊 Final Analysis for: {hdf5_path}")

    obj_positions = []

    try:
        with h5py.File(hdf5_path, "r") as f:
            demos = list(f["data"].keys())
            if not demos:
                print("❌ HDF5 file is empty.")
                return

            first_demo = demos[0]
            if "obs" not in f[f"data/{first_demo}"]:
                print("❌ 'obs' group not found.")
                return

            # --- 1. 自动寻找 Key ---
            available_keys = list(f[f"data/{first_demo}/obs"].keys())
            candidates = [
                "privileged_target_pos",  # 新数据优先
                "SquareNut_main_pose",
                "object",  # 旧数据通用 Key
                "object_pose"
            ]

            target_key = None
            for cand in candidates:
                if cand in available_keys:
                    target_key = cand
                    break

            if target_key is None:
                print(f"❌ Could not auto-detect object key. Available: {available_keys}")
                return

            print(f"🔑 Using Key: [{target_key}]")
            print(f"🔢 Total Demos: {len(demos)}")

            # --- 2. 提取数据 ---
            for demo_key in demos:
                obs = f["data"][demo_key]["obs"]
                if target_key in obs:
                    data = obs[target_key][0]
                    # 强制只取前3维 (x,y,z)，忽略 quaternion 或其他拼接信息
                    pos = data[:3]
                    # 简单的零点过滤
                    if np.linalg.norm(pos) > 1e-6:
                        obj_positions.append(pos)

    except Exception as e:
        print(f"❌ Error reading file: {e}")
        return

    if not obj_positions:
        print("❌ No valid positions found.")
        return

    points = np.array(obj_positions)

    print(f"\n--- 📦 Object Distribution Quality ---")

    # 指标 1: 全局散布
    avg_pairwise = np.mean(pdist(points))
    print(f"1️⃣  Avg Pairwise Dist:            {avg_pairwise:.4f} m")

    # 指标 2: 空间覆盖率
    grid_size = 0.02
    voxel_indices = np.floor(points / grid_size).astype(int)
    unique_voxels = np.unique(voxel_indices, axis=0)
    occupied_count = len(unique_voxels)
    efficiency = occupied_count / len(points)

    print(f"2️⃣  Grid Coverage (2cm grids):    {occupied_count} grids")
    print(f"3️⃣  Sampling Efficiency:          {efficiency:.2%}")

    # 指标 3: 分布范围 (Bounding Box)
    min_xyz = np.min(points, axis=0)
    max_xyz = np.max(points, axis=0)
    range_xyz = max_xyz - min_xyz
    print(f"4️⃣  Range X: {range_xyz[0]:.4f}m | Y: {range_xyz[1]:.4f}m | Z: {range_xyz[2]:.4f}m")

    # 可视化
    plt.figure(figsize=(8, 8))

    # 强制画 X-Y 平面，因为那是桌面
    x = points[:, 0]
    y = points[:, 1]

    plt.scatter(x, y, alpha=0.6, c='crimson', edgecolors='k', s=40, label='Object Pos')

    # 自动调整坐标轴范围，使其等比例显示
    plt.axis('equal')
    plt.grid(True, alpha=0.3)

    plt.title(f"Object Distribution (Top-Down X-Y)\nKey: {target_key}")
    plt.xlabel("X (m)")
    plt.ylabel("Y (m)")
    plt.legend()

    save_name = f"diversity_plot_{target_key}_{index}.png"
    plt.savefig(save_name)
    print(f"\n✅ Plot saved to {save_name}")
    print("   -> 请打开这张图，看看点是不是排成了一条线？")


if __name__ == "__main__":
    # 分别运行两次看看对比
    print("\n--- Analysing OLD Data ---")
    analyze_diversity_final("/home/zgz/projects/second_work/cpgen/datasets/debug/metrics/merged_demos_obs.hdf5", 1)

    print("\n--- Analysing NEW Data ---")
    analyze_diversity_final("/home/zgz/projects/second_work/cpgen/datasets/debug/metrics/merged_demos_obs_2.hdf5", 2)