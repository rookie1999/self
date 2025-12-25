import h5py
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import ConvexHull
from sklearn.decomposition import PCA


def analyze_diversity_enhanced(hdf5_path, object_name_key="cube_pose"):  # 替换成你 obs 里实际的物体 key
    print(f"{'=' * 20}\nAnalyzing diversity for: {hdf5_path}")

    eef_positions = []
    obj_positions = []

    with h5py.File(hdf5_path, "r") as f:
        demos = list(f["data"].keys())
        print(f"Total Demos: {len(demos)}")

        for demo_key in demos:
            # 1. 获取 Robot EEF Pos (第0帧)
            if "obs" in f["data"][demo_key]:
                obs = f["data"][demo_key]["obs"]

                # 获取末端位置
                if "robot0_eef_pos" in obs:
                    eef_positions.append(obs["robot0_eef_pos"][0])

                # 2. 获取 Object Pos (第0帧) - 关键修改
                # 请根据你的 hdf5 结构调整 key，比如 'object', 'cube_pos', 'object_pose' 等
                # 通常 pose 是 7维 (x,y,z,qx,qy,qz,qw)，我们取前3位
                if object_name_key in obs:
                    obj_pos = obs[object_name_key][0]
                    # 如果是pose (7维)，取前3；如果是pos (3维)，取全部
                    obj_positions.append(obj_pos[:3] if len(obj_pos) > 3 else obj_pos)
            else:
                pass

    # 封装一个内部函数来计算指标，避免重复代码
    def compute_metrics(name, points_list):
        if not points_list:
            print(f"[{name}] No data found.")
            return

        points = np.array(points_list)
        print(f"\n--- {name} Analysis ---")

        # 标准差
        std_dev = np.std(points, axis=0)
        print(f"Std Dev (X, Y, Z): {std_dev}")

        # 平均距离
        from scipy.spatial.distance import pdist
        avg_dist = np.mean(pdist(points))
        print(f"Avg Pairwise Dist: {avg_dist:.5f} m")

        # 凸包体积
        try:
            hull = ConvexHull(points)
            print(f"Convex Hull Vol:   {hull.volume:.5f} m^3")
        except:
            print("Convex Hull: N/A (Points coplanar or insufficient)")

        return points

    # 执行分析
    pts_eef = compute_metrics("Robot EEF", eef_positions)
    pts_obj = compute_metrics("Object", obj_positions)

    # 3. 可视化对比 (如果有物体数据)
    if pts_obj is not None:
        plt.figure(figsize=(12, 5))

        # 绘制 EEF 分布
        plt.subplot(1, 2, 1)
        plt.scatter(pts_eef[:, 0], pts_eef[:, 1], alpha=0.5, c='blue', label='EEF')
        plt.title("Robot EEF Start Positions (X-Y)")
        plt.xlabel("X");
        plt.ylabel("Y");
        plt.grid(True);
        plt.legend()

        # 绘制 Object 分布
        plt.subplot(1, 2, 2)
        plt.scatter(pts_obj[:, 0], pts_obj[:, 1], alpha=0.5, c='red', label='Object')
        plt.title("Object Start Positions (X-Y)")
        plt.xlabel("X");
        plt.ylabel("Y");
        plt.grid(True);
        plt.legend()

        plt.tight_layout()
        plt.savefig("diversity_comparison_obj.png")
        print("\nPlot saved to diversity_comparison_obj.png")
    else:
        print("\n[Warning] No object data found using key:", object_name_key)
        print(
            "Please check your HDF5 structure (use h5py_viewer or simple print script) to find the correct key for object position.")


if __name__ == "__main__":
    # 替换这里
    file_path = "/home/zgz/projects/second_work/cpgen/datasets/debug/metrics/merged_demos_obs.hdf5"
    analyze_diversity_enhanced(file_path, object_name_key="object")