import json

import h5py


h5_path = "/home/benson/projects/second_work/cpgen/datasets/generated/Coffee_D0/2025-12-05-14-58-55/successes/14-59-36-848113.hdf5"

with h5py.File(h5_path, 'r') as f:
    # 假设你要读取名为 'dataset_name' 的数据集
    dataset_name = 'data'

    if dataset_name in f:
        ds = f[dataset_name]
        print(f"\n[数据集 '{dataset_name}' 的属性]:")
        for key, value in ds.attrs.items():
            print(f"  {key}: {value}")
    else:
        print(f"找不到数据集: {dataset_name}")

    demo_group = f[f"data/demo_0"]
    # print(demo_group.attrs["model_file"])  # MJCF文档
    # print("Start to print constraint data:")
    # print(demo_group["constraint_data"])
    print(list(demo_group.keys()))
    print("Start to print controller config:")
    env_args_str = f["data"].attrs["env_args"]
    env_args = json.loads(env_args_str)
    print(env_args["env_kwargs"]["controller_configs"])
    print("over")
