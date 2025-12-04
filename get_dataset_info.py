import h5py

with h5py.File('/home/benson/projects/second_work/cpgen/datasets/source/coffee.hdf5', 'r') as f:
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
    print("Start to print constraint data:")
    print(demo_group["constraint_data"])