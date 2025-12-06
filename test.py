import copy
from typing import List

import numpy as np
import tyro

from demo_aug.generate import set_seed, Demo, load_demos, Config, CPEnvRobomimic, Constraint, ConstraintGenerator


def visualize(env):
    print(">>> 开始可视化环境 (按 Ctrl+C 退出)...")
    try:
        # 重置环境
        obs = env.reset()
        if hasattr(env, "action_dimension"):
            act_dim = env.action_dimension
        elif hasattr(env, "action_space"):
            act_dim = env.action_space.shape[0]
        else:
            # 最后的兜底：Panda 机械臂通常是 7 自由度 + 1 夹爪 = 8，或者 7
            act_dim = 7

        for i in range(10000):
            # 生成一个全零动作或者随机动作让环境运行
            # action = np.zeros(env.action_dim)
            action = np.random.randn(act_dim) * 0.1  # 给一点微小的随机动作看效果

            # 执行动作
            obs, r, done, info = env.step(action)

            # 渲染画面 (robosuite 环境通常在 step 中会自动调用 render，但显式调用更保险)
            env.render()
    except KeyboardInterrupt:
        print("停止可视化")


def main(cfg: Config):
    set_seed(cfg.seed)
    np.set_printoptions(suppress=True, precision=4)
    src_demos: List[Demo] = load_demos(
        cfg.demo_path,
        start_idx=cfg.load_demos_start_idx,
        end_idx=cfg.load_demos_end_idx,
    )
    # create the environment
    import robomimic.utils.env_utils as EnvUtils
    import robomimic.utils.file_utils as FileUtils
    import robomimic.utils.obs_utils as ObsUtils

    # note: above import doesn't work; need update robosuite repo's osc itself to assume world frame actions

    # Remove after OSC in abs world frame merged into config refactoring
    dummy_spec = dict(
        obs=dict(
            low_dim=["robot0_eef_pos"],
            rgb=["agentview_image", "agentview"],
        ),
    )
    # 根据映射表将每个数据的名字与类型进行匹配
    ObsUtils.initialize_obs_utils_with_obs_specs(obs_modality_specs=dummy_spec)
    # 从HDF5文件中读取当初录制数据的环境配置，即env_args
    env_meta = FileUtils.get_env_metadata_from_dataset(dataset_path=cfg.demo_path)
    # update controller config to use abs actions
    # 这里将控制器调整成绝对位置控制，为了与之后的运动规划器计算的坐标相匹配
    env_meta["env_kwargs"]["controller_configs"]["control_delta"] = False
    # 由于使用运动规划，不需要看图，所以use_camera_obs为False
    # 但是debug的时候需要保存为video进行检查轨迹是否正确，需要has_offscreen_renderer为True
    # viz_robot_kpts表示是否需要可视化机器人关键点，在debug下为True
    if cfg.debug:
        env_meta["env_kwargs"]["use_camera_obs"] = False
        env_meta["env_kwargs"]["has_offscreen_renderer"] = True
        viz_robot_kpts = True
    else:
        env_meta["env_kwargs"]["use_camera_obs"] = False
        env_meta["env_kwargs"]["has_offscreen_renderer"] = False
        viz_robot_kpts = False

    from robosuite.controllers import load_composite_controller_config

    if cfg.controller_type == "default":
        controller_config = load_composite_controller_config(robot="Panda")
        controller_config["body_parts"]["right"]["input_type"] = "absolute"
        controller_config["body_parts"]["right"]["input_ref_frame"] = "world"
    elif cfg.controller_type == "ik":
        controller_config = load_composite_controller_config(
            controller="demo_aug/configs/robosuite/panda_ik.json"
        )
    env_meta["env_kwargs"]["controller_configs"] = controller_config
    # 配置环境重置时候的扰动
    if cfg.initialization.initialization_noise_type is not None:
        env_meta["env_kwargs"]["initialization_noise"] = {
            "type": cfg.initialization.initialization_noise_type,  # 通常是“gaussian”或者“uniform”，决定了抖动的概率分布
            "magnitude": cfg.initialization.initialization_noise_magnitude,  # 决定了抖动的幅度
        }
    if "env_name" in env_meta["env_kwargs"]:  # 防止参数冲突，env_name是suite.make的第一个参数，而不是通过这里的kwargs传入的
        del env_meta["env_kwargs"]["env_name"]
    src_env_meta = copy.deepcopy(env_meta)
    # src_env前缀，表示source environment。它是只读的，用来回放原始的演示数据。
    # 为了渲染（图像处理））
    # src_env_w_rendering这个带有图像的环境只是为了提取出关键点
    src_env_w_rendering = (
        EnvUtils.create_env_from_metadata(  # used for rendering only segmented demos
            env_meta=src_env_meta,
            use_image_obs=True,
            render_offscreen=True,
            # render=False,
            render=True,
        )
    )

    src_env = EnvUtils.create_env_from_metadata(  # not great that we need to specify these keys
        env_meta=src_env_meta,
        use_image_obs=False,
        render_offscreen=False,
        render=True,
    )

    src_env = CPEnvRobomimic(src_env)
    env_meta["env_name"] = cfg.env_name

    import robosuite
    env_meta["env_version"] = robosuite.__version__
    # env是交互环境，用来跑新的仿真。
    env = EnvUtils.create_env_from_metadata(  # not great that we need to specify these keys
        env_meta=env_meta,
        use_image_obs=False,
        render_offscreen=False,
        render=True,
    )

    env = CPEnvRobomimic(env)
    constraint_sequences: List[List[Constraint]] = ConstraintGenerator(
        src_env,
        demos=src_demos,
        target_env=env,
        src_env_w_rendering=src_env_w_rendering,
        override_constraints=cfg.override_constraints,
        override_interactions=cfg.override_interactions,
        custom_constraints_path=cfg.custom_constraints_path,
        demo_segmentation_type=cfg.demo_segmentation_type,
    ).generate_constraints()
    visualize(env)



    original_xml = env.env.env.sim.model.get_xml()
    if "box" in original_xml:  # 或者你修改的特定名称
        print(">>> 成功加载了新的环境模型 (Box)!")
    else:
        print(">>> 警告: 环境模型似乎仍是旧版本 (Coffee)!")

if __name__ == "__main__":
    tyro.cli(main)