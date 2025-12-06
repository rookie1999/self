"""
To test generated dataset, run the following command:

python robomimic/scripts/playback_dataset.py --dataset ../demo-aug/datasets/generated/2024-10-10.hdf5  --video_path test.mp4
"""

import copy
import logging
import os
from types import SimpleNamespace

from demo_aug.utils.demo_segmentation_utils import (
    create_constraint,
    decompose_trajectory,
    parse_interactions,
    run_llm_e2e_segmentation,
    run_llm_success_segmentation,
    unparse_interactions,
)

os.environ["MUJOCO_GL"] = "egl"

import json
import math
import pathlib
import random
import re
import subprocess
import threading
import time
from collections import defaultdict
from dataclasses import asdict, dataclass, field
from datetime import datetime
from typing import Any, Callable, Dict, List, Literal, Optional, Sequence, Tuple, Union

import cpgen_envs as cpgen_envs
import h5py
import imageio
import mujoco
import numpy as np
import robosuite
import torch
import tyro
from lxml import etree as ET
from mink import Configuration
from mujoco import MjData, MjModel, mj_fwdPosition, viewer

# monkey patch robomimic's EnvBase.reset_to
from robomimic.envs.env_robosuite import EnvRobosuite
from scipy.optimize import dual_annealing, minimize
from scipy.spatial.transform import Rotation as R
from tqdm import tqdm

import demo_aug
import wandb
from demo_aug import transform_keypoints
from demo_aug.envs.motion_planners.base_mp import MotionPlanner
from demo_aug.envs.motion_planners.curobo_mp import (
    CuroboMotionPlanner,  # Use lxml instead of xml.etree.ElementTree
)
from demo_aug.envs.motion_planners.eef_interp_curobo_mp import (
    EEFInterpCuroboMotionPlanner,
)
from demo_aug.envs.motion_planners.eef_interp_mp import (
    EEFInterpMinkMotionPlanner,
    EEFInterpMotionPlanner,
)
from demo_aug.envs.motion_planners.indexed_configuration import IndexedConfiguration
from demo_aug.utils.file_utils import count_total_demos, merge_demo_files
from demo_aug.utils.logging_utils import setup_file_logger
from demo_aug.utils.mathutils import make_pose, random_pose

# simulation framework  # mainly for version checking at the moment
from demo_aug.utils.mujoco_utils import (
    check_geom_collisions,
    get_body_name,
    get_geom_names,
    get_joint_name_to_indexes,
    get_subtree_geom_ids_by_group,
    get_top_level_bodies,
    render_image,
    set_body_pose,
    sync_mjdata,
    sync_mjmodel_mjdata,
    update_fixed_joint_objects_in_xml,
)
from demo_aug.utils.xml_utils import (
    add_free_joint,
    remove_actuator_tag,
    remove_arm_keep_eef,
    update_xml_with_mjmodel,
)


def set_seed(seed: int) -> None:
    random.seed(seed)  # Python's random module
    np.random.seed(seed)  # Numpy seed
    torch.manual_seed(seed)  # PyTorch seed for CPU
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)  # PyTorch seed for all CUDA devices

    # Ensure reproducibility for CUDA algorithms if desired (optional, comes with performance hit)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


@dataclass
class Demo:
    """
    states与observations是不同的。
    states包含了物理世界上帝视角的所有数据，如机器人关节角度、桌子上物体的坐标+旋转角度+速度
    observations包含了机器人能够感知到的数据，如机器人本体数据（这一部分确实与states的部分数据重合），视觉数据
    """
    demo_path: str
    states: np.ndarray
    actions: np.ndarray
    observations: Dict[str, Any]
    demo_num: Optional[int] = None
    # 具体可以参见Constraint class
    constraint_data_sequence: Optional[List[Any]] = (
        None  # need to shift Constraint class up
    )
    model_file: Optional[str] = None  # specific to mujoco
    env_args: Optional[Dict[str, Any]] = None  # 调用robosuite.make时所需的所有参数及其配置
    # demo中发生的关键交互事件（Contact Events）的描述
    # 一个字符串或列表，格式类似于 实体1:实体2
    interactions: Optional[Union[str, Sequence[Union[str, Tuple[str, str]]]]] = None


def update_demo_interactions(demo_path: str, demo_idx: int, interactions: str) -> None:
    """Update the interactions attribute for a specified demo at runtime."""
    with h5py.File(demo_path, "a") as f:
        demo_group = f[f"data/demo_{demo_idx}"]
        demo_group.attrs["interactions"] = interactions
    logging.info(f"Updated interactions for demo_{demo_idx} in {demo_path}")


def recursively_unpack_h5py(obj) -> Dict[str, Any]:
    """将一个 HDF5 对象（通常是存储在硬盘上的、层级复杂的结构）完全读取到内存中，并转换成 Python 原生的 字典 (dict) 和 NumPy 数组"""
    if isinstance(obj, h5py.Group):
        return {key: recursively_unpack_h5py(obj[key]) for key in obj.keys()}
    elif isinstance(obj, h5py.Dataset):
        data = obj[()]
        # HDF5 和 Python 对“字符串”的编码格式不兼容
        # HDF5: 喜欢存 Bytes (字节串)，比如 b'hello'
        # Python: 喜欢用 Unicode (字符串)，比如 'hello'
        if isinstance(data, np.ndarray):
            if data.dtype.kind == "S":  # Binary string
                return np.char.decode(data, "utf-8")
            elif data.dtype.kind == "O":  # Object array, possibly containing strings
                return np.array(
                    [
                        item.decode("utf-8") if isinstance(item, bytes) else item
                        for item in data
                    ]
                )
        elif isinstance(data, bytes):  # Single binary string
            return data.decode("utf-8")
        return data
    else:
        return obj


def load_demos(
    demo_path: str, start_idx: int = 0, end_idx: Optional[int] = None
) -> List[Demo]:
    """
    从hdf5文件中按照指定的索引范围批量读取演示数据，
    内容包括状态、动作、高级的约束信息constraints、交互信息interactions、以及环境配置envs
    """
    demos: List[Demo] = []
    with h5py.File(demo_path, "r") as f:
        src_demos = f["data"]
        if end_idx is None or end_idx == -1:
            end_idx = len(src_demos)
        for idx in range(start_idx, end_idx):
            demo_group = f[f"data/demo_{idx}"]
            constraint_data_sequence = None
            if "constraint_data" in demo_group:
                constraint_data_sequence = []
                for constraint_data in demo_group["constraint_data"].values():
                    # constraint_data是一个0维的json字符串，通过[()]这样的方式可以直接拿到其内容
                    cdict = json.loads(constraint_data[()])
                    constraint_data_sequence.append(
                        Constraint.from_constraint_data_dict(cdict)
                    )
            # Load interactions as a string if present.
            interactions = demo_group.attrs.get("interactions")
            if interactions is not None:
                interactions = str(interactions)
            demo = Demo(
                demo_path=demo_path,
                states=np.array(demo_group["states"]),
                actions=np.array(demo_group["actions"]),
                observations=recursively_unpack_h5py(demo_group["obs"])
                if "obs" in demo_group
                else {},
                demo_num=idx,
                constraint_data_sequence=constraint_data_sequence,
                model_file=demo_group.attrs["model_file"],
                env_args=json.loads(f["data"].attrs["env_args"]),
                interactions=interactions,
            )
            demos.append(demo)
    return demos


@dataclass
class PlanningResult:
    actions: List[np.ndarray]
    success: bool
    robot_configurations: Optional[List[np.ndarray]] = None


class CPEnv:
    def __init__(self, env=None):
        self.env = env
        # Additional initialization code for CPEnv
        possible_task_relevant_obj_names = []
        excluded_obj_keywords = {
            "target",
            "robot",
            "mount",
            "gripper",
            "world",
            "table",
        }
        model = self.env.env.sim.model._model
        for i in range(model.nbody):
            name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, i)
            if name and not any(keyword in name for keyword in excluded_obj_keywords):
                possible_task_relevant_obj_names.append(name)
        self.possible_task_relevant_obj_names = possible_task_relevant_obj_names

    def step(self, action: np.ndarray) -> Tuple[Dict, float, bool, Dict]:
        obs, reward, done, info = self.env.step(action)
        self._update_obs(obs)
        return obs, reward, done, info

    @staticmethod
    def is_collision(
        model: mujoco.MjModel,
        data: mujoco.MjData,
        robot_body_name: str = "gripper0_right_right_gripper",
        exclude_prefixes: List[str] = ["robot", "gripper"],
        verbose: bool = False,
    ) -> bool:
        """
        Check if there is a collision between the robot and non-robot geometries.

        Parameters:
            env: The environment containing the simulation model and data.
            robot_body_name: The name of the robot body to check collisions for.
            exclude_prefixes: Prefixes for bodies to exclude from non-robot geoms.
            verbose: Whether to log additional information.

        Returns:
            collided (bool): True if a collision is detected, False otherwise.
        """
        # Get robot geometries
        robot_geoms = get_subtree_geom_ids_by_group(
            model, model.body(robot_body_name).id, group=0
        )

        # Get non-robot geometries
        body_ids = get_top_level_bodies(model, exclude_prefixes=exclude_prefixes)
        body_names = [get_body_name(model, body_id) for body_id in body_ids]
        non_robot_geoms = [
            geom_id
            for body_id in body_ids
            for geom_id in get_subtree_geom_ids_by_group(model, body_id, group=0)
        ]

        if verbose:
            logging.info(f"Top-level body names: {body_names}")
            logging.info(f"Filtered top-level bodies: {body_ids}")

        # Check for collisions between robot and non-robot geometries
        geom_pairs_to_check: List[Tuple] = [(robot_geoms, non_robot_geoms)]
        collided = len(check_geom_collisions(model, data, geom_pairs_to_check)) > 0

        return collided

    def reset(self, let_env_settle: bool = True) -> Tuple[Dict, float, bool, Dict]:
        obs = self.env.reset()
        while CPEnv.is_collision(
            self.env.env.sim.model._model, self.env.env.sim.data._data
        ):
            print(
                "reset caused collision, re-trying ..."
            )  # TODO: technically robosuite should handle
            obs = self.env.reset()
        self._update_obs(obs)
        obs = self.let_env_settle()
        return obs

    def let_env_settle(self, num_steps: int = 6) -> Dict:
        action = np.zeros(7)
        action[:3] = self.env.get_observation()["robot0_eef_pos"]
        action[3:6] = quat2axisangle(self.env.get_observation()["robot0_eef_quat_site"])
        for _ in range(num_steps):
            obs, _, _, _ = self.env.step(action)
            self._update_obs(obs)
        return obs

    def get_observation(self) -> Dict:
        # update observations
        self.env.env._update_observables(force=True)
        obs = self.env.get_observation()
        self._update_obs(obs)
        return obs

    def get_obj_geoms_size(self, obj_name: str) -> Dict[str, np.ndarray]:
        # Code to get object size
        return {}

    def _update_obs(self, obs: Dict):
        for obj_name in self.possible_task_relevant_obj_names:
            obs[obj_name + "_pose"] = self.get_obj_pose(obj_name)
            obs[obj_name + "_geoms_size"] = self.get_obj_geoms_size(obj_name)

        obs["robot_q"] = np.concatenate(
            [obs["robot0_joint_pos"], obs["robot0_gripper_qpos"]]
        )
        arm = self.env.env.robots[0].arms[0]
        obs["robot0_eef_pos_body"] = self.env.env.sim.data.get_body_xpos(
            self.env.env.robots[0].robot_model.eef_name[arm]
        )

    def get_collision_geometry(self):
        # Code to get collision geometry
        pass

    def get_robot_configuration(self) -> np.ndarray:
        # Code to get robot configuration
        pass

    def get_obj_pose(self, obj_name: str) -> np.ndarray:
        # Code to get object pose
        pass

    def get_obj_geom_transform(self, obj_name: str) -> Union[np.ndarray, Callable]:
        """
        Get the object geometry transform. We can assume the transform is applied to the object's mesh if applicable.
        """
        # Code to get object geometry transform
        pass

    def update_pose_frame(
        self, src_frame: str, pose_src_frame: np.ndarray, dest_frame: str
    ) -> np.ndarray:
        # Code to get pose
        pass

    # from robosuite's wrapper.py
    def __getattr__(self, attr):
        # using getattr ensures that both __getattribute__ and __getattr__ (fallback) get called
        # (see https://stackoverflow.com/questions/3278077/difference-between-getattr-vs-getattribute)
        orig_attr = getattr(self.env, attr)
        if callable(orig_attr):

            def hooked(*args, **kwargs):
                result = orig_attr(*args, **kwargs)
                # prevent wrapped_class from becoming unwrapped
                if result is self.env:
                    return self
                return result

            return hooked
        else:
            return orig_attr


class CPEnvRobomimic(CPEnv):
    def get_obj_pose(self, obj_name: str) -> np.ndarray:
        """
        Get the pose of an object in the environment.

        Args:
            obj_name (str): Name of the object.

        Returns:
            np.ndarray: 4x4 transformation matrix of the object.
        """
        pose = np.eye(4)
        body_id = self.env.env.sim.model.body_name2id(obj_name)
        pose[:3, :3] = self.env.env.sim.data.body_xmat[body_id].reshape(3, 3)
        pose[:3, 3] = self.env.env.sim.data.body_xpos[body_id]
        return pose

    def set_obj_pose(self, obj_name: str, pose: np.ndarray) -> None:
        """
        Set the pose of an object in the environment.

        Args:
            obj_name (str): Name of the object.
            pose (np.ndarray): 4x4 transformation matrix of the object.
        """
        body_id = self.env.env.sim.model.body_name2id(obj_name)
        joint_id = self.env.env.sim.model.body_jntadr[body_id]
        # get join tname
        joint_name = self.env.env.sim.model.joint_id2name(joint_id)
        # set the joint of the robot
        pos = pose[:3, 3]
        quat_xyzw = mat2quat(pose[:3, :3])
        pos_quat = np.concatenate([pos, quat_xyzw])
        self.env.env.sim.data.set_joint_qpos(joint_name, pos_quat)

    def get_obj_geoms_size(self, obj_name: str) -> Dict[str, np.ndarray]:
        body_id = self.env.env.sim.model.body_name2id(obj_name)
        geom_ids = get_subtree_geom_ids_by_group(
            self.env.env.sim.model, body_id, group=0
        ) + get_subtree_geom_ids_by_group(self.env.env.sim.model, body_id, group=1)
        geom_names = get_geom_names(self.env.env.sim.model._model, geom_ids)
        geom_names = [
            geom_name if geom_name != "" else f"g{i}"
            for i, geom_name in enumerate(geom_names)
        ]

        sizes = {}
        for geom_id, geom_name in zip(geom_ids, geom_names):
            size = self.env.env.sim.model.geom_size[geom_id]
            sizes[geom_name] = size
        return sizes

    def update_pose_frame(
        self, src_frame: str, pose_src_frame: np.ndarray, dest_frame: str
    ) -> np.ndarray:
        """
        Given a source frame and pose in src_frame, return the pose in the dest frame.

        Args:
            src_frame: Name of the source frame.
            pose_src_frame: Pose in the source frame (4x4 transformation matrix).
            dest_frame: Name of the destination frame.

        Returns:
            Pose in the destination frame (4x4 transformation matrix).
        """
        X_W_src = self.get_obj_pose(src_frame)
        X_W_dst = self.get_obj_pose(dest_frame)
        X_src_obj = pose_src_frame

        # Transform the source frame pose to world coordinates
        X_W_obj = np.dot(X_W_src, X_src_obj)
        X_dst_obj = np.linalg.inv(X_W_dst) @ X_W_obj
        return X_dst_obj


class SquareEnv(CPEnvRobomimic):
    def _update_obs(self, obs: Dict):
        obs["SquareNut_main_pose"] = self.get_obj_pose("SquareNut_main")
        obs["peg1_pose"] = self.get_obj_pose("peg1")
        obs["robot_q"] = np.concatenate(
            [obs["robot0_joint_pos"], obs["robot0_gripper_qpos"]]
        )


class ThreePieceAssemblyEnv(CPEnvRobomimic):
    def _update_obs(self, obs: Dict):
        obs["piece_1_root_pose"] = self.get_obj_pose("piece_1_root")
        obs["base_root_pose"] = self.get_obj_pose("base_root")
        obs["piece_2_root_pose"] = self.get_obj_pose("piece_2_root")
        obs["robot_q"] = np.concatenate(
            [obs["robot0_joint_pos"], obs["robot0_gripper_qpos"]]
        )


class MugCleanupEnv(CPEnvRobomimic):
    def _update_obs(self, obs: Dict):
        obs["DrawerObject_main_pose"] = self.get_obj_pose("DrawerObject_main")
        obs["cleanup_object_main_pose"] = self.get_obj_pose("cleanup_object_main")
        obs["robot_q"] = np.concatenate(
            [obs["robot0_joint_pos"], obs["robot0_gripper_qpos"]]
        )


# @dataclass会自动为下面的类生成 __init__（构造函数）、__repr__（打印格式）等常用方法。你不需要自己写 self.type = type 这种赋值代码了，它主要用于存储数据。
@dataclass
class Symmetry:
    # 表示对称性应用在哪个对象上，这里有三个"obj0","obj1","robot-eef-xyz"
    type: Literal["obj0", "obj1", "obj0", "obj1", "robot-eef-xyz"]
    # 定义了 transforms 属性，这是一个列表，里面存的是具体的变换矩阵。
    # 它描述了具体的对称变换方式。比如“绕 Z 轴旋转 180 度”。
    transforms: Optional[List[Dict[str, np.ndarray]]] = (
        None  # e.g. 0 degrees and 180 degrees; use LLM or something
    )
    # 如果为 False：算法会使用原始演示中的位姿 加上 transforms 里定义的额外位姿。
    # 如果为 True：算法会忽略原始演示中的位姿，只使用 transforms 里定义的位姿。这通常用于原始演示的角度不太好，想强制修正的时候。
    skip_default: bool = False  # whether to skip the default non-symmetry transform
    # effectively changes the source demo; hacky code --- better to update the source demo itself ...

    def to_dict(self):
        """
        Serialize this symmetry into a dictionary for saving/JSON/etc.
        """
        return {
            "type": self.type,
            "transforms": self.transforms,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Symmetry":
        """Reconstruct a Symmetry object from a dict."""
        return cls(
            type=data["type"],
            transforms=data.get("transforms", None),
            skip_default=data.get("skip_default", False),
        )


@dataclass
class Constraint:
    obj_names: Optional[List[str]] = None  # 参与这个约束的物体名称列表
    src_obj_pose: Optional[Dict[str, np.ndarray]] = None  # 源演示中，每一帧物体的位姿（4x4 变换矩阵）
    src_obs: Optional[Dict[str, np.ndarray]] = None  # 源演示中的观测值（传感器数据）和物理状态（上帝视角数据）
    src_obj_transform: Optional[Dict[str, Union[np.ndarray, Callable]]] = None  #
    src_obj_geoms_size: Optional[Dict[str, Dict[str, np.ndarray]]] = None  # 源演示中，物体的物理尺寸（长宽高/半径等）
    src_action: Optional[np.ndarray] = None  # 源演示中机器人当时采取的动作（关节速度或末端位移）
    src_state: Optional[np.ndarray] = None  # MuJoCo 的完整物理状态向量（Simulation State），包含所有关节的角度 (qpos)、所有关节的速度 (qvel)、致动器的状态等。
    src_model_file: Optional[str] = None  # specific to mujoco：MuJoCo 的 XML 模型文件内容。用于完全复现当时的物理参数
    src_gripper_action: Optional[np.ndarray] = None  # 源演示中机器人当时采取的夹爪动作（开/合）
    # 在物体坐标系下，关键点在哪里
    keypoints_obj_frame_annotation: Optional[Dict[str, np.ndarray]] = field(
        default_factory=dict
    )  # obj_name -> keypoints_obj_frame
    # 机器人连杆坐标系下，关键点在哪里
    keypoints_robot_link_frame_annotation: Optional[Dict[str, np.ndarray]] = field(
        default_factory=dict
    )
    # 随时间变化的数据。记录了在演示的每一帧中，这些关键点在物体坐标系下的位置
    # 规划器的目标就是让“当前的机器人关键点”去重合“演示中的物体关键点”
    keypoints_obj_frame: Optional[Dict[str, np.ndarray]] = field(
        default_factory=dict
    )  # obj_name -> keypoints_obj_frame
    # 随时间变化的数据。记录了在演示的每一帧中，这些关键点在机器人连杆坐标系下的位置
    keypoints_robot_link_frame: Optional[Dict[str, np.ndarray]] = (
        None  # link_name -> keypoints_link_frame
    )
    symmetries: List[Symmetry] = field(default_factory=list)
    # 是否镜像末端执行器。这通常用于处理对称抓取（比如从左边抓和从右边抓是一样的）。
    reflect_eef: bool = True
    # None: 默认行为，通过运动规划跟随轨迹。
    # "grasp": 抓紧。将物体固定在手上，防止滑落。
    during_constraint_behavior: Optional[Literal["grasp"]] = None
    # 包含多个后续动作
    #    "lift": 提起来。在约束结束后，执行一个垂直向上的运动。这是为了防止物体在移动前还在桌面上摩擦，或者作为任务完成的标志。
    #    "open_gripper": 张开夹爪。通常用于“放置”任务完成后，松开物体。
    post_constraint_behavior: List[Literal["lift", "open_gripper"]] = field(
        default_factory=list
    )
    # 是否重复动作。可能用于某些需要更高控制频率或慢速执行的场景。
    duplicate_actions: bool = False
    name: Optional[str] = "no-name-constraint"  # 约束的名称
    # 定义约束的物理拓扑关系
    #     robot-object：机器人与物体的约束。通常指“抓取”或“推动”。意味着机器人的末端执行器（EEF）需要与物体保持某种相对位置关系。
    #     object-object：物体与物体的约束。通常指“放置”、“插入”或“堆叠”。例如把方块放在桌子上，或者把螺母拧在螺栓上。
    constraint_type: Literal["robot-object", "object-object"] = "robot-object"
    # 焊接关系。定义物体挂在哪个父物体下面
    # 如果值为 {"Cube": "gripper_link"}，说明方块现在被视为“焊”在夹爪上，随夹爪运动
    obj_to_parent_attachment_frame: Optional[Dict[str, Optional[str]]] = None
    timesteps: List[int] = field(default_factory=list)  # 约束在原始Demo中生效的时间步索引列表
    # 如果为 True，在开始生成新演示时，不要总是从第 0 帧开始，而是随机从这个约束中间的某一帧开始。这能增加数据的多样性。
    reset_near_random_constraint_state: bool = False
    # 重置时的位置噪声范围。例如 0.15 表示在重置时，可以在演示位置的基础上增加 ±15cm 的随机抖动。这是为了训练出的策略具有鲁棒性（抗干扰能力）。
    reset_near_constraint_pos_bound: float = 0.15
    """For reset-near-constraint. If True, reset env to random timestep inside one of timesteps."""

    def to_constraint_data_dict(self) -> Dict[str, Any]:
        """
        Returns a dictionary containing only the fields that are *not* inferred or
        completed by `_complete_constraint` in ConstraintGenerator. This makes the dictionary lightweight and
        suitable for serialization (e.g., saving to an HDF5 file).
        """
        symmetries_serialized = [s.to_dict() for s in self.symmetries]

        return {
            "name": self.name,
            "constraint_type": self.constraint_type,
            "obj_names": self.obj_names,
            "timesteps": self.timesteps,
            "keypoints_obj_frame_annotation": self.keypoints_obj_frame_annotation,
            "keypoints_robot_link_frame_annotation": self.keypoints_robot_link_frame_annotation,
            "symmetries": symmetries_serialized,
            "during_constraint_behavior": self.during_constraint_behavior,
            "post_constraint_behavior": self.post_constraint_behavior,
            "obj_to_parent_attachment_frame": self.obj_to_parent_attachment_frame,
            "reset_near_random_constraint_state": self.reset_near_random_constraint_state,
            "src_model_file": self.src_model_file,
            "duplicate_actions": self.duplicate_actions,
        }

    # 这里的@property的作用就是，通过self.attached_object_names直接访问，而不是通过函数调用的方式，且是只读的
    @property
    def attached_object_names(self) -> List[str]:
        """返回那些处于“被抓取状态”或“被连接状态”的物体名称列表"""
        if self.obj_to_parent_attachment_frame is None:
            return []
        return [
            obj_name
            for obj_name, parent_frame in self.obj_to_parent_attachment_frame.items()
            if parent_frame is not None
        ]

    @classmethod
    def from_constraint_data_dict(cls, data: Dict[str, Any]) -> "Constraint":
        """
        Reconstructs a Constraint object from a "lightweight" constraint-data dict.
        This is effectively the reverse of `to_constraint_data_dict()`.
        """

        # 1) Convert list-of-dicts to list of Symmetry objects:
        symmetries = [Symmetry.from_dict(sym) for sym in data.get("symmetries", [])]

        # 2) Convert Python lists back to np.ndarray for any keypoint data
        keypoints_robot = data.get("keypoints_robot_link_frame_annotation", {})
        keypoints_obj = data.get("keypoints_obj_frame_annotation", {})

        return cls(
            name=data.get("name"),
            constraint_type=data["constraint_type"],
            obj_names=data.get("obj_names"),
            timesteps=data.get("timesteps", []),
            keypoints_robot_link_frame_annotation=keypoints_robot,
            keypoints_obj_frame_annotation=keypoints_obj,
            symmetries=symmetries,
            reflect_eef=data.get("reflect_eef", True),
            during_constraint_behavior=data.get("during_constraint_behavior"),
            post_constraint_behavior=data.get("post_constraint_behavior", []),
            obj_to_parent_attachment_frame=data.get("obj_to_parent_attachment_frame"),
            reset_near_random_constraint_state=data.get(
                "reset_near_random_constraint_state", False
            ),
            src_model_file=data.get("src_model_file", None),
            duplicate_actions=data.get("duplicate_actions", False),
        )

    # consider other case where there's a weld constraint, though that's probably handled by previous stuff already?
    def keypoints_world_frame(
        self,
        obj_name: str,
        obj_pose: np.ndarray,
        obj_transform: Union[np.ndarray, Callable],
        robot_pose: np.ndarray,
    ) -> np.ndarray:
        """将物体坐标系下的关键点转换到世界坐标系下"""
        # 物体被“抓”在机器人手上（Attached/Welded） 当机器人抓住一个物体时，物理引擎或者代码逻辑会将这个物体视为机器人的一部分。
        keypoints_obj_frame = self.keypoints_obj_frame[obj_name]
        keypoints_world_frame = np.zeros_like(keypoints_obj_frame)
        for i, keypoint in enumerate(keypoints_obj_frame):
            # np.append(keypoint, 1)：将关键点坐标 (x, y, z) 变成齐次坐标 (x, y, z, 1)，以便进行矩阵乘法。
            # 算出了关键点相对于物体中心的精确位置
            transformed_keypoint = np.dot(obj_transform, np.append(keypoint, 1))
            # 处理纯平移
            # transformed_keypoint[:3]：点相对于物体的位置。
            # obj_pose[:3]：**物体相对于父级（这里隐含是机器人）**的位置。
            # robot_pose[:3]：父级（机器人）相对于世界的位置。
            keypoints_world_frame[i] = (
                transformed_keypoint[:3] + obj_pose[:3] + robot_pose[:3]
            )
        return keypoints_world_frame

    # 没有用到
    def is_satisfied(
        self, obj_name: str, obj_pose: np.ndarray, obj_state: np.ndarray
    ) -> bool:
        """Check if the constraint is satisfied based on the object state."""
        # Implement the logic to check if the constraint is satisfied
        # using the provided object state (pose and transform)
        return False

    def update_constraint(self, pose, transform, task_relevant_obj):
        if task_relevant_obj in self.task_relevant_objects:
            self.pose = pose
            self.transform = transform


def is_env_state_close(
    env: CPEnv,
    constraint: Constraint,
    pos_threshold=0.01,
    rot_threshold=0.1,
    eef_pos_threshold=0.01,
    eef_rot_threshold=0.05,
    gripper_threshold=0.01,
):
    """
    Checks if the current environment state is close enough to the stored src_obj_pose.

    Args:
        env: The environment instance providing observations.
        constraint: A Constraint object containing object names and reference poses.
        pos_threshold: Position difference threshold in meters.
        rot_threshold: Rotation difference threshold in radians.
        eef_pos_threshold: EEF position difference threshold (meters).
        eef_rot_threshold: EEF rotation difference threshold (radians).
        gripper_threshold: Gripper qpos difference threshold.
    Returns:
        bool: True if all task-relevant object poses are within the threshold.
    """
    if constraint.obj_names is None or constraint.src_obj_pose is None:
        logging.info("is_env_state_close: no relevant obs")
        return False  # No reference to compare

    obs = env.get_observation()  # Assuming env provides a method to get observations
    for obj_name in constraint.obj_names:
        if (
            f"{obj_name}_pose" not in obs.keys()
            or obj_name not in constraint.src_obj_pose
        ):
            continue  # Skip if object pose isn't available

        current_pose = obs[f"{obj_name}_pose"]  # 4x4 pose matrix
        ref_pose = constraint.src_obj_pose[obj_name][-1]  # 4x4 stored pose matrix

        # Compute translation difference
        pos_diff = np.linalg.norm(current_pose[:3, 3] - ref_pose[:3, 3])

        # Compute rotation difference using Frobenius norm
        rot_diff = np.linalg.norm(current_pose[:3, :3] - ref_pose[:3, :3], ord="fro")

        if pos_diff > pos_threshold or rot_diff > rot_threshold:
            logging.info(
                f"is_env_state_close: object: {obj_name} pos_diff: {pos_diff} rot_diff: {rot_diff}"
            )
            return False  # If any object exceeds the threshold, return False

    if constraint.src_obs is None:
        return False

    # Compare EEF pose
    if "robot0_eef_pos" in obs and "robot0_eef_pos" in constraint.src_obs.keys():
        current_eef_pos = obs["robot0_eef_pos"]  # 3x1 matrix
        ref_eef_pos = constraint.src_obs["robot0_eef_pos"][-1]  # 3x1 stored matrix
        current_eef_quat = obs["robot0_eef_quat_site"]
        ref_eef_quat = constraint.src_obs["robot0_eef_quat_site"][
            -1
        ]  # 3x1 stored matrix

        eef_pos_diff = np.linalg.norm(current_eef_pos - ref_eef_pos)
        eef_rot_diff = np.linalg.norm(current_eef_quat - ref_eef_quat)

        if eef_pos_diff > eef_pos_threshold or eef_rot_diff > eef_rot_threshold:
            logging.info(
                f"is_env_state_close: eef_pos_diff > eef_pos_threshold {eef_pos_diff} > {eef_pos_threshold}"
            )
            logging.info(
                f"is_env_state_close: eef_rot_diff > eef_rot_threshold {eef_rot_diff} > {eef_rot_threshold}"
            )
            return False

    # Compare gripper joint positions
    if (
        "robot0_gripper_qpos" in obs
        and "robot0_gripper_qpos" in constraint.src_obs.keys()
    ):
        current_gripper_qpos = np.array(
            obs["robot0_gripper_qpos"]
        )  # Ensure it's an array
        ref_gripper_qpos = constraint.src_obs["robot0_gripper_qpos"][-1]

        gripper_diff = np.linalg.norm(current_gripper_qpos - ref_gripper_qpos)

        if gripper_diff > gripper_threshold:
            logging.info(
                f"is_env_state_close: gripper_diff > gripper_threshold {gripper_diff} > {gripper_threshold}"
            )
            return False

    return True  # All objects are within the threshold


def reset_to(self, state, no_return_obs: bool = False):
    """
    Reset to a specific simulator state.

    Args:
        state (dict): current simulator state that contains one or more of:
            - states (np.ndarray): initial state of the mujoco environment
            - model (str): mujoco scene xml
        no_return_obs (bool): if True, do not return observation after setting the simulator state.
            Used to not waste computation when we don't need the observation.
            If False, return observation after setting the simulator state.
    Returns:
        observation (dict): observation dictionary after setting the simulator state (only
            if "states" is in @state)
    """
    should_ret = False
    if "model" in state:
        self.reset()
        robosuite_version_id = int(robosuite.__version__.split(".")[1])
        if robosuite_version_id <= 3:
            from robosuite.utils.mjcf_utils import postprocess_model_xml

            xml = postprocess_model_xml(state["model"])
        else:
            # v1.4 and above use the class-based edit_model_xml function
            xml = self.env.edit_model_xml(state["model"])

        from robosuite.utils.binding_utils import MjSim

        # first try to reset using MjSim
        try:
            MjSim.from_xml_string(xml)
            self.env.reset_from_xml_string(xml)
        except Exception as e:
            logging.info(f"Error in reset_to: {e}")
        self.env.sim.reset()
        if not self._is_v1:
            # hide teleop visualization after restoring from model
            self.env.sim.model.site_rgba[self.env.eef_site_id] = np.array(
                [0.0, 0.0, 0.0, 0.0]
            )
            self.env.sim.model.site_rgba[self.env.eef_cylinder_id] = np.array(
                [0.0, 0.0, 0.0, 0.0]
            )
    if "states" in state:
        self.env.sim.set_state_from_flattened(state["states"])
        self.env.sim.forward()
        should_ret = True

    if "goal" in state:
        self.set_goal(**state["goal"])
    if not no_return_obs and should_ret:
        # only return obs if we've done a forward call - otherwise the observations will be garbage
        return self.get_observation()
    return None


EnvRobosuite.reset_to = reset_to


def get_constraint_data(
    demo: Demo,
    src_env,
    override_constraints: bool = False,
    override_interactions: bool = False,
    demo_segmentation_type: Optional[Literal[
        "distance-based", "llm-e2e", "llm-success"
    ]] = None,
    interaction_threshold: float = 0.03,
    create_video: bool = False,
    constraints_path: Optional[str] = None,
) -> List[Constraint]:
    """
    Extracts constraints from a demo using the provided source environment.
    create_video (bool): Whether to create a video of the dataset for visualization.
        Doesn't always works due to old dataset being from old robosuite version (pre-1.5); may
        mess up later env resets due to how mujoco models work.
    constraints_path (str): Path to save constraints to. If None, saves to the default path.
    demo_segmentation_type:
        distance-based:计算机器人末端（手）和目标物体之间的欧几里得距离。
            * 如果距离大于某个阈值（例如 interaction_threshold = 0.03），系统认为机器人在“移动 (Motion)”。
            * 如果距离小于阈值，并且 interactions 列表里记录了这两个物体有互动，系统认为机器人在“交互 (Constraint)”（比如正在抓取）。
        llm-e2e:把演示过程的视频帧（图像）或者文本描述喂给大语言模型（LLM）。LLM 像人类一样，通过视觉语义来理解动作（"End-to-End" 意味着直接输出分段结果）。
        llm-success:不是让 LLM 通篇看视频切分，而是让 LLM 检查**“关键子任务是否成功完成”。
            * 系统可能会把视频按时间窗喂给 LLM，并不断问：“现在的方块已经被抓起来了吗？(Success?)”
            * 当 LLM 回答“是”的瞬间，系统就标记这里是上一阶段（抓取）的结束点，下一阶段（移动）的开始点。

    Assume src_env comes from demo.
    """
    constraints: List[Constraint] = []
    # .stem属性：获取文件名（不包含扩展名/后缀）。
    default_constraints_path = (
        (
            pathlib.Path(demo.demo_path).parent
            / pathlib.Path(demo.demo_path).stem
            / (f"{demo_segmentation_type}-constraints.json" if demo_segmentation_type else "constraints.json")
        )
        if constraints_path is None
        else pathlib.Path(constraints_path)
    )
    default_segments_output_dir = (
        pathlib.Path(demo.demo_path).parent
        / pathlib.Path(demo.demo_path).stem
        / (f"{demo_segmentation_type}-segments" if demo_segmentation_type else "segments")
    )
    default_video_dir = (
        pathlib.Path(demo.demo_path).parent
        / pathlib.Path(demo.demo_path).stem
        / "videos"
    )
    # Ensure necessary directories exist
    default_constraints_path.parent.mkdir(parents=True, exist_ok=True)
    default_segments_output_dir.mkdir(parents=True, exist_ok=True)
    default_video_dir.mkdir(parents=True, exist_ok=True)

    # Create video of dataset if visualization needed
    # 如果create_video且输出目录下没有mp4文件，则需要将抽象的数据渲染成视频
    if create_video and (
        not default_video_dir.exists() or not any(default_video_dir.glob("*.mp4"))
    ):
        # 用于把物理状态渲染成图片
        from robomimic.scripts.dataset_states_to_obs import dataset_states_to_obs

        from scripts.dataset.mp4_from_h5 import Config as MP4H5Config
        # 用于把图片合成视频
        from scripts.dataset.mp4_from_h5 import generate_videos_from_hdf5

        render_camera_names = ["agentview", "robot0_eye_in_hand"]
        camera_height, camera_width = 128, 128
        depth = False
        # SimpleNamespace是一个空壳对象，主要是可以通过访问属性的方式得到值，效果与dict类似
        dataset_states_to_obs_args = SimpleNamespace(
            dataset=demo.demo_path,
            output_name=f"{str(pathlib.Path(demo.demo_path).stem)}_obs.hdf5",
            n=None,
            shaped=False,
            camera_names=render_camera_names,
            camera_height=camera_height,
            camera_width=camera_width,
            depth=depth,
            done_mode=1,  # done = done or (t == traj_len)
            copy_rewards=False,
            copy_dones=False,
            exclude_next_obs=True,
            compress=False,
            use_actions=False,
        )
        # 执行渲染。这步最耗时，它会跑 MuJoCo 仿真，把每一帧的状态变成图片存入新的 HDF5。
        dataset_states_to_obs(dataset_states_to_obs_args)
        # 生成的带图片的 HDF5的路径
        dataset_with_obs_path = (
            pathlib.Path(demo.demo_path).parent / dataset_states_to_obs_args.output_name
        )
        # 转化为视频
        # ax_posix方法是将路径中的反斜杠变成正斜杠
        generate_videos_from_hdf5(
            MP4H5Config(
                h5_file_path=dataset_with_obs_path,
                all_demos=True,
                output_dir=default_video_dir.as_posix(),
                fps=20,
                show_time_idx=True,
            )
        )

    # 如果约束文件不存在，或者强制覆盖，则需要重新计算约束
    if (not default_constraints_path.exists()) or override_constraints:
        segments = []
        if demo.interactions is None or override_interactions:
            # 提示用户输入interactions
            resp = input(
                f"Type in interactions as a list of object interaction tuples 'entity1:entity2'. "
                f"For robot-object interactions, prefix the robot entity with 'gripper0_right_right_gripper' "
                f"(e.g., 'gripper0_right_right_gripper:object'). "
                f"You can refer to the video at {default_video_dir.as_posix()} for reference. "
            )
            assert (
                demo.demo_num is not None
            ), "demo_num must be set to update interactions in demo.demo_path"
            update_demo_interactions(demo.demo_path, demo.demo_num, resp)
            interactions = parse_interactions(resp)
            demo.interactions = interactions
        if isinstance(demo.interactions, str):
            demo.interactions = parse_interactions(demo.interactions)
        # include first model file to actually get the original; also, reset_to that file.
        # 重置环境。将 src_env 的物理模型（XML）重置为演示数据中记录的那个 XML。
        src_env.reset_to({"model": demo.model_file})
        # segments的形式如：[
        #     ("motion", [0, 40]),
        #     ("robot-object", [41, 100]),
        #     ...
        # ]
        # (label, time_range)这里的time_range是[start, end]
        if demo_segmentation_type == "distance-based":
            # decompose trajectory into segments
            segments = decompose_trajectory(
                src_env,
                demo.states,
                interaction_threshold,
                demo.interactions,
                default_segments_output_dir,
            )
        elif demo_segmentation_type == "llm-e2e":
            segments = run_llm_e2e_segmentation(
                src_env,
                demo.states,
                lang_description=None,
                interactions=unparse_interactions(demo.interactions),
                segments_output_dir=default_segments_output_dir,
            )
        elif demo_segmentation_type == "llm-success":
            segments = run_llm_success_segmentation(
                src_env,
                demo.states,
                lang_description=None,
                interactions=unparse_interactions(demo.interactions),
                segments_output_dir=default_segments_output_dir,
            )
        # 创建非移动的约束
        constraints_data = [
            create_constraint(
                label,
                time_range,
                gripper_type=src_env.env.robots[0].gripper["right"].__class__.__name__,
            )
            for label, time_range in segments
            if label != "motion"
        ]
        with open(default_constraints_path, "w") as f:
            json.dump(constraints_data, f, indent=4)

    with open(default_constraints_path, "r") as f:
        constraints_data = json.load(f)
        for constraint_data in constraints_data:
            constraints.append(Constraint.from_constraint_data_dict(constraint_data))

    return constraints


def adjust_constraint_timesteps_for_collision_free_motion(
    constraints: List[Dict],
    env,
    states: np.ndarray,
    robot_body_name: str = "robot",
    verbose: bool = False,
    robot_weld_frame: str = "gripper0_right_eef",
) -> List[Dict]:
    """Adjusts the first and last timesteps of constraints to ensure they are collision-free.

    Args:
        constraints (List[Dict]): List of constraint dictionaries.
        env: MuJoCo environment object supporting reset_to({"states": state}).
        model: MuJoCo model object.
        states (np.ndarray): Array of environment states corresponding to timesteps.
        robot_body_name (str, optional): Root body name of the robot. Defaults to "robot".
        verbose (bool, optional): Whether to log debug info. Defaults to False.

    Returns:
        List[Dict]: Updated constraints with adjusted timestep ranges.
    """
    model = env.env.env.sim.model._model
    data = env.env.env.sim.data._data

    # Get robot geometries
    robot_geoms = get_subtree_geom_ids_by_group(
        model, model.body(robot_body_name).id, group=0
    )

    for i, constraint in enumerate(constraints):
        # Determine current constraint welded objects (by attachment frame)
        welded_bodies = set()
        if "obj_to_parent_attachment_frame" in constraint:
            for obj, parent in constraint["obj_to_parent_attachment_frame"].items():
                if (
                    parent == robot_weld_frame
                ):  # Assumption: welded objects are attached here
                    welded_bodies.add(obj)

        # Exclude default prefixes (always ignore collisions with these)
        default_excludes = ["robot", "gripper", "table"]
        # (They will be concatenated with welded objects later.)

        timesteps = constraint.get("timesteps", [])
        if not timesteps:
            continue

        # Adjust the start timestep: move earlier until collision free.
        new_start_idx = timesteps[0]
        while new_start_idx >= 0:
            env.reset_to({"states": states[new_start_idx]})
            # Check collisions between robot and non-robot bodies.
            body_ids = get_top_level_bodies(
                model, exclude_prefixes=default_excludes + list(welded_bodies)
            )
            non_robot_geoms = [
                geom_id
                for body_id in body_ids
                for geom_id in get_subtree_geom_ids_by_group(model, body_id, group=0)
            ]
            geom_pairs_to_check: List[Tuple] = [(robot_geoms, non_robot_geoms)]
            if (
                len(
                    check_geom_collisions(
                        model, data, geom_pairs_to_check, collision_activation_dist=0.03
                    )
                )
                == 0
            ):
                break
            new_start_idx -= 1

        # Adjust the last timestep: move later until collision free.
        new_end_idx = timesteps[-1]
        while new_end_idx < len(states) - 1:
            env.reset_to({"states": states[new_end_idx]})
            # For object-object constraints, add extra welded objects.
            object_object_welded = set()
            if constraint.get("constraint_type") == "object-object":
                object_object_welded.update(constraint.get("obj_names", []))
            # Build local exclusion set based on current and next constraint.
            current_attached = set(welded_bodies)
            if i < len(constraints) - 1:
                next_constraint = constraints[i + 1]
                next_welded = set()
                if "obj_to_parent_attachment_frame" in next_constraint:
                    for obj, parent in next_constraint[
                        "obj_to_parent_attachment_frame"
                    ].items():
                        if parent == robot_weld_frame:
                            next_welded.add(obj)
                    # If next constraint involves a welded object, add it.
                    exclude_set = current_attached.union(object_object_welded).union(
                        next_welded
                    )
                else:
                    # Otherwise, exclude only the originally attached bodies.
                    exclude_set = current_attached
            else:
                exclude_set = current_attached.union(object_object_welded)
            # Combine with default exclusions.
            exclude_prefixes = default_excludes + list(exclude_set)

            body_ids = get_top_level_bodies(model, exclude_prefixes=exclude_prefixes)
            non_robot_geoms = [
                geom_id
                for body_id in body_ids
                for geom_id in get_subtree_geom_ids_by_group(model, body_id, group=0)
            ]
            geom_pairs_to_check = [(robot_geoms, non_robot_geoms)]
            if len(check_geom_collisions(model, data, geom_pairs_to_check)) == 0:
                break
            new_end_idx += 1

        # Ensure new_end_idx is valid.
        new_end_idx = max(new_start_idx, new_end_idx)
        constraint["timesteps"] = list(range(new_start_idx, new_end_idx + 1))
        if verbose:
            print(f"original start and end timesteps: {timesteps[0], timesteps[-1]}")
            print(f"new start and end timesteps: {new_start_idx, new_end_idx}")
    return constraints


class ConstraintGenerator:
    """充当“翻译官”的角色，将原始的演示数据（Demo）批量转换成机器人可以执行的几何约束序列（Constraint Sequences）。"""
    def __init__(
        self,
        env,  # 录制或生成演示数据（Demos）时的原始环境实例
        demos: List[Demo],
        target_env: Optional[CPEnv] = None,  # 生成的约束最终要被执行或测试的环境
        src_env_w_rendering=None,  # 带渲染功能的源环境 (Source Environment with Rendering)
        override_constraints: bool = False,
        override_interactions: bool = False,
        custom_constraints_path: Optional[str] = None,
        demo_segmentation_type: Optional[Literal["distance-based", "llm-e2e"]] = None,
    ):
        """
        Constraint generator for generating constraints from source demos.
        """
        self.env = env
        self.demos = demos
        self.target_env = target_env  # if target_env is None, it's the same as env
        self.src_env_w_rendering = src_env_w_rendering
        self.override_constraints = override_constraints
        self.override_interactions = override_interactions
        self.custom_constraints_path = custom_constraints_path
        self.demo_segmentation_type = demo_segmentation_type

    def generate_constraints(self) -> List[List[Constraint]]:
        """遍历所有的演示数据（Demo），为每一个 Demo 生成对应的一系列几何约束。"""
        constraint_sequences: List[List[Constraint]] = []
        for demo in tqdm(self.demos, desc="Getting constraints for source demos"):
            if demo.constraint_data_sequence is None:
                constraint_data_lst = get_constraint_data(
                    demo,
                    self.src_env_w_rendering,
                    override_constraints=self.override_constraints,
                    override_interactions=self.override_interactions,
                    constraints_path=self.custom_constraints_path,
                    demo_segmentation_type=self.demo_segmentation_type,
                )
            else:
                constraint_data_lst = demo.constraint_data_sequence
            constraint_sequence = []
            for constraint_data in constraint_data_lst:
                # adjust_constraint_timesteps_for_collision_free_motion(
                #     constraints=[constraint_data],
                #     env=self.env,
                #     states=demo.states,
                #     robot_body_name="robot0_base",
                #     verbose=False,
                # )
                # 将constraint_data元数据（比如由大模型生成的文本描述，或者简单的关键帧索引），变成代码中可执行的 Constraint 对象
                constraint: Constraint = self._complete_constraint(
                    constraint_data, demo, self.env
                )
                constraint_sequence.append(constraint)
            constraint_sequences.append(constraint_sequence)
        return constraint_sequences

    def _complete_constraint(
        self,
        constraint: Constraint,
        demo: Demo,
        env: CPEnv,
        target_env: Optional[CPEnv] = None,
    ) -> Constraint:
        """
        启动物理仿真器，时光倒流回到演示发生的每一个时刻，精确计算机器人、物体之间的相对位置关系，并将所有几何信息打包填入 Constraint 对象中。
        Complete the constraint datastructure using additional information from the environment and source demo.
        """
        timesteps = constraint.timesteps
        object_names = constraint.obj_names
        constraint_type = constraint.constraint_type

        keypoints_robot_link_frame = copy.deepcopy(
            constraint.keypoints_robot_link_frame_annotation
        )
        keypoints_obj_frame = copy.deepcopy(constraint.keypoints_obj_frame_annotation)
        src_obj_pose: Dict[str, np.ndarray] = defaultdict(list)  # 物体在每一帧的位姿
        src_obs: Dict[str, List[Dict[str, np.ndarray]]] = defaultdict(list)  # 每一帧的观测数据
        src_obj_geoms_size: Dict[str, List[Dict[str, np.ndarray]]] = defaultdict(list)  # 物体的几何尺寸（用于碰撞检测）
        src_obj_transform: Dict[str, Union[np.ndarray, Callable]] = {}

        if (
            demo.env_args is not None
            and demo.env_args.get("env_version", None) is not None
            and demo.env_args["env_version"] == robosuite.__version__
        ):
            # 不仅重置状态，还重新加载 XML 模型文件
            env.env.reset_to({"states": demo.states[0], "model": demo.model_file})
            # TODO: maybe do sync_fixed_joint_objects_in_mjmodel except inverse i.e. get values from xml and update mjmodel
            # unfortunately, resetting model file is really slow;
            # +1 reason for using original source demo + annotations
            # only reset-to model file if robosuite version matches env args; else may fail
            # some fixed objects' poses are only reflected in the model file;
            # TODO: have a better way to specify fixed objects' locations, currently specific to robosuite
        if constraint_type == "robot-object":
            # Expand robot keypoints across time
            for robot_frame, keypoints_t_kpts_xyz in keypoints_robot_link_frame.items():
                keypoints_robot_link_frame[robot_frame] = np.repeat(
                    keypoints_t_kpts_xyz, len(timesteps), axis=0
                )

            # Calculate object keypoints
            object_keypoints = keypoints_obj_frame[object_names[0]] = [
                [] for _ in range(len(timesteps))
            ]
            for timestep_idx, timestep in enumerate(timesteps):
                env.env.reset_to({"states": demo.states[timestep]})
                object_name = object_names[0]
                for (
                    robot_link,
                    keypoints_robot_frame,
                ) in keypoints_robot_link_frame.items():
                    for keypoint in keypoints_robot_frame[timestep_idx]:
                        keypt_pose_src_frame = np.eye(4)
                        keypt_pose_src_frame[:3, 3] = keypoint
                        pose_obj_frame = env.update_pose_frame(
                            src_frame=robot_link,
                            pose_src_frame=keypt_pose_src_frame,
                            dest_frame=object_name,
                        )
                        object_keypoints[timestep_idx].append(pose_obj_frame[:3, 3])

                object_keypoints[timestep_idx] = np.array(
                    object_keypoints[timestep_idx]
                )

        elif constraint_type == "object-object":
            # Expand object keypoints across time
            for object_name, keypoints_t_kpts_xyz in keypoints_obj_frame.items():
                keypoints_obj_frame[object_name] = np.repeat(
                    keypoints_t_kpts_xyz, len(timesteps), axis=0
                )

            # Initialize empty keypoints for other objects
            for object_name in object_names:
                if object_name not in keypoints_obj_frame:
                    keypoints_obj_frame[object_name] = [
                        [] for _ in range(len(timesteps))
                    ]

            # Calculate keypoints for second object
            for timestep_idx, timestep in enumerate(timesteps):
                env.env.reset_to({"states": demo.states[timestep]})
                assert keypoints_obj_frame.get(object_names[0]) is not None, (
                    f"Keyframes for object {object_names[0]} not found. "
                    "Ensure that the first object in the constraint has keypoints."
                )
                keypoints_obj_frame_first = keypoints_obj_frame[object_names[0]]
                for keypoint in keypoints_obj_frame_first[timestep_idx]:
                    keypt_pose_src_frame = np.eye(4)
                    keypt_pose_src_frame[:3, 3] = keypoint
                    object_name = object_names[1]
                    object_keypoint_pose = env.update_pose_frame(
                        src_frame=object_names[0],
                        pose_src_frame=keypt_pose_src_frame,
                        dest_frame=object_name,
                    )
                    keypoints_obj_frame[object_name][timestep_idx].append(
                        object_keypoint_pose[:3, 3]
                    )

                keypoints_obj_frame[object_name][timestep_idx] = np.array(
                    keypoints_obj_frame[object_name][timestep_idx]
                )

        # Collect source actions and observations
        src_action = np.array([demo.actions[t] for t in timesteps])
        src_gripper_action = src_action[:, -1:]

        for t in timesteps:
            obs = env.env.reset_to({"states": demo.states[t]})
            for obj_name in object_names:
                src_obj_pose[obj_name].append(env.get_obj_pose(obj_name))
                src_obj_transform[obj_name] = env.get_obj_geom_transform(obj_name)
                src_obj_geoms_size[obj_name].append(env.get_obj_geoms_size(obj_name))
            copy_obs = ["robot0_gripper_qpos", "robot0_eef_pos", "robot0_eef_quat_site"]
            for obs_name in copy_obs:
                src_obs[obs_name].append(obs[obs_name])

        src_state = np.array([demo.states[t] for t in timesteps])

        # Update constraint with collected data
        constraint.src_obj_pose = {
            obj_name: np.array(obj_pose) for obj_name, obj_pose in src_obj_pose.items()
        }
        constraint.src_obs = src_obs
        constraint.src_obj_transform = src_obj_transform
        constraint.src_obj_geoms_size = src_obj_geoms_size
        constraint.src_action = src_action
        constraint.src_state = src_state
        constraint.src_model_file = demo.model_file
        constraint.src_gripper_action = src_gripper_action
        constraint.keypoints_robot_link_frame = keypoints_robot_link_frame
        constraint.keypoints_obj_frame = keypoints_obj_frame
        return constraint


EPS = np.finfo(float).eps * 4.0


# @jit_decorator
def quat2mat(quaternion):
    """
    将四元数转换成旋转矩阵
    Converts given quaternion to matrix.
    Args:
        quaternion (np.array): (x,y,z,w) vec4 float angles
    Returns:
        np.array: 3x3 rotation matrix
    """
    # awkward semantics for use with numba
    # 从[x,y,z,w] 转换成 [w,x,y,z]
    inds = np.array([3, 0, 1, 2])
    q = np.asarray(quaternion).copy().astype(np.float32)[inds]

    n = np.dot(q, q)
    if n < EPS:
        # 如果四元数的模长非常接近 0（是一个零四元数），它无法表示旋转，且后面无法做除法运算。
        # 直接返回单位矩阵
        return np.identity(3)
    q *= math.sqrt(2.0 / n)
    q2 = np.outer(q, q)
    return np.array(
        [
            [1.0 - q2[2, 2] - q2[3, 3], q2[1, 2] - q2[3, 0], q2[1, 3] + q2[2, 0]],
            [q2[1, 2] + q2[3, 0], 1.0 - q2[1, 1] - q2[3, 3], q2[2, 3] - q2[1, 0]],
            [q2[1, 3] - q2[2, 0], q2[2, 3] + q2[1, 0], 1.0 - q2[1, 1] - q2[2, 2]],
        ]
    )


# 正常轴角是有两个独立的量组成的，一个是旋转轴，一个是旋转角
# 这里用三个量来表示，vec = theta * n。这里theta是vec方向，n是vec的模长。
def quat2axisangle(quat):
    """
    四元数转轴角
    Converts quaternion to axis-angle format.
    Returns a unit vector direction scaled by its angle in radians.

    Args:
        quat (np.array): (x,y,z,w) vec4 float angles

    Returns:
        np.array: (ax,ay,az) axis-angle exponential coordinates
    """
    # clip quaternion
    if quat[3] > 1.0:
        quat[3] = 1.0
    elif quat[3] < -1.0:
        quat[3] = -1.0

    den = np.sqrt(1.0 - quat[3] * quat[3])
    if math.isclose(den, 0.0):
        # This is (close to) a zero degree rotation, immediately return
        return np.zeros(3)

    return (quat[:3] * 2.0 * math.acos(quat[3])) / den


def axisangle2quat(vec):
    """
    轴角转四元数
    Converts scaled axis-angle to quat.

    Args:
        vec (np.array): (ax,ay,az) axis-angle exponential coordinates

    Returns:
        np.array: (x,y,z,w) vec4 float angles
    """
    # Grab angle
    angle = np.linalg.norm(vec)

    # handle zero-rotation case
    if math.isclose(angle, 0.0):
        return np.array([0.0, 0.0, 0.0, 1.0])

    # make sure that axis is a unit vector
    axis = vec / angle

    q = np.zeros(4)
    q[3] = np.cos(angle / 2.0)
    q[:3] = axis * np.sin(angle / 2.0)
    return q


###########################################################################


def visualize_robot_configuration(
    robot_configuration: Configuration, env: CPEnv, q: np.ndarray
):
    robot_configuration.update(q)
    env.env.env.robots[0].set_robot_joint_positions(q[:-2])
    env.env.env.robots[0].set_gripper_joint_positions(q[-2:])
    env.env.env.viewer.update()


def visualize_robot_configuration_and_keypoints(
    robot_configuration: Configuration,
    env: CPEnv,
    q: np.ndarray,
    constraint: Constraint,
):
    robot_configuration.update(q)
    env.env.env.robots[0].set_robot_joint_positions(q[:-2])
    env.env.env.robots[0].set_gripper_joint_positions(q[-2:])
    kpts = robot_configuration.get_keypoints()
    max_mocaps = min(env.env.env.sim.data.mocap_pos.shape[0], len(kpts))
    env.env.env.sim.data.mocap_pos[:max_mocaps] = kpts[:max_mocaps]

    if constraint.constraint_type == "object-object":
        # get pose from robot_configuration and set env pose for viz
        obj_name = constraint.obj_names[0]
        pose = robot_configuration.get_transform_frame_to_world(
            obj_name, "body"
        ).as_matrix()
        env.set_obj_pose(constraint.obj_names[0], pose)

    env.env.env.sim.forward()
    env.env.env.viewer.update()


def set_mocap_pos_and_update_viewer(
    env: CPEnv, mocap_pos: np.ndarray, update_viewer: bool = True
):
    max_mocaps = min(env.env.env.sim.data.mocap_pos.shape[0], len(mocap_pos))
    env.env.env.sim.data.mocap_pos[:max_mocaps] = mocap_pos[:max_mocaps]
    env.env.env.sim.forward()
    if update_viewer:
        env.env.env.viewer.update()


############################
# XML manipulation methods #
############################


def mat2quat(rmat: np.ndarray) -> np.ndarray:
    """
    Returns:
        np.array: (w, x,y,z) float quaternion angles
    """
    return np.roll(R.from_matrix(rmat).as_quat(), 1)


def remove_free_joints(xml_string: str) -> str:
    # Parse the XML string
    root = ET.fromstring(xml_string)

    # Find all bodies with free joints
    bodies_with_free_joints = root.xpath(".//body[joint[@type='free']]")

    if not bodies_with_free_joints:
        print("No bodies with free joints found.")
        return xml_string

    # Removes free joints
    for body in bodies_with_free_joints:
        joints = body.xpath(
            "./joint[@type='free']"
        )  # Find the free joints within the body
        for joint in joints:
            body.remove(joint)  # Remove the free joint from the body

    # Convert the tree back to an XML string
    return ET.tostring(root, encoding="unicode")


def weld_frames(
    xml_string: str, frame_on_parent_F: str, frame_on_child_M: str, X_FM: np.ndarray
) -> str:
    """
    在 XML 文本层面进行“移花接木”的手术。
    它将一个物体（Child）从原来的位置剪切下来，去掉它的自由度（关节），然后以固定的相对位姿（Pose）“粘贴”到另一个物体（Parent）下面。
    """
    root = ET.fromstring(xml_string)
    # Find the parent frame body
    parent_body = root.find(f".//body[@name='{frame_on_parent_F}']")
    if parent_body is None:
        raise ValueError(f"Parent frame '{frame_on_parent_F}' not found")

    # Find the child frame body
    child_body = root.find(f".//body[@name='{frame_on_child_M}']")
    if child_body is None:
        raise ValueError(f"Child frame '{frame_on_child_M}' not found")

    # Remove the child frame from its original location
    # 把子物体从它原来的层级结构中拿出来
    for parent in root.iter():
        if child_body in list(parent):
            parent.remove(child_body)
            break

    # Remove the existing free joint from the child frame (if exists)
    # 去掉自由物体对应的free_joint
    free_joint = child_body.find(".//joint")
    if free_joint is not None:
        child_body.remove(free_joint)

    # Extract position and quaternion from X_FM
    # 从传入的变换矩阵 X_FM 中提取平移向量 (position) 和旋转矩阵（转为四元数 quaternion）。
    position = X_FM[:3, 3]
    quaternion = mat2quat(X_FM[:3, :3])

    # Create a new pos attribute for the child frame
    # 设置子物体的位置
    pos_str = f"{position[0]:.6f} {position[1]:.6f} {position[2]:.6f}"
    child_body.set("pos", pos_str)

    # Create a new quat attribute for the child frame
    # 设置子物体的的quat属性
    quat_str = f"{quaternion[0]:.6f} {quaternion[1]:.6f} {quaternion[2]:.6f} {quaternion[3]:.6f}"
    child_body.set("quat", quat_str)

    # Append the child frame to the parent frame
    parent_body.append(child_body)

    # Convert the modified XML tree back to a string
    modified_xml_string = ET.tostring(root, encoding="unicode")
    return modified_xml_string


def attach_square_nut_to_gripper(xml_string: str, X_gripper_object: np.ndarray) -> str:
    # Parse the XML string
    root = ET.fromstring(xml_string)

    # Find the gripper0_right_eef body
    gripper_body = root.find(".//body[@name='gripper0_right_eef']")
    if gripper_body is None:
        raise ValueError("Gripper body not found")

    # Find the SquareNut_main body
    square_nut_body = root.find(".//body[@name='SquareNut_main']")

    if square_nut_body is None:
        raise ValueError("Square nut body not found")

    # Remove the SquareNut_main body from its original location
    for parent in root.iter():
        if square_nut_body in list(parent):
            parent.remove(square_nut_body)
            break

    # Remove the existing free joint from the SquareNut_main body
    free_joint = square_nut_body.find(".//joint")
    if free_joint is not None:
        square_nut_body.remove(free_joint)

    # Extract position and quaternion from X_gripper_object
    position = X_gripper_object[:3, 3]
    quaternion = mat2quat(X_gripper_object[:3, :3])

    # Create a new pos attribute for the SquareNut_main body
    pos_str = f"{position[0]:.6f} {position[1]:.6f} {position[2]:.6f}"
    square_nut_body.set("pos", pos_str)

    # Create a new quat attribute for the SquareNut_main body
    quat_str = f"{quaternion[0]:.6f} {quaternion[1]:.6f} {quaternion[2]:.6f} {quaternion[3]:.6f}"
    square_nut_body.set("quat", quat_str)

    # Append the SquareNut_main body to the gripper0_right_eef body
    gripper_body.append(square_nut_body)

    # Convert the modified XML tree back to a string
    modified_xml_string = ET.tostring(root, encoding="unicode")
    return modified_xml_string


def create_passive_viewer(
    model: MjModel,
    data: MjData,
    rate: float = 0.5,
    run_physics: bool = False,
    robot_configuration: Optional[Configuration] = None,
    eef_configuration: Optional[Configuration] = None,
    eef_pose: Optional[np.ndarray] = None,
    **kwargs: Any,
) -> None:
    """
    Create a passive viewer for the given MuJoCo model and data.
    Args:
        model (MjModel): The MuJoCo model object.
        data (MjData): The MuJoCo data object.
        rate (float): The update rate for the viewer (in seconds).
        run_physics (bool): Whether to run physics in the background.
        **kwargs: Additional keyword arguments.
    """
    with viewer.launch_passive(
        model=model, data=data, show_left_ui=False, show_right_ui=False
    ) as vis:
        mj_fwdPosition(model, data)  # Initialize positions
        qs = kwargs.get("qs", None)
        i: int = 0
        while vis.is_running():
            if run_physics:
                mujoco.mj_step(model, data)
            if eef_configuration is not None:
                assert eef_pose is not None, "eef_pose is required."
                xyz_wxyz = np.concatenate([eef_pose[:3, 3], mat2quat(eef_pose[:3, :3])])
                original_q = data.qpos.copy()
                original_q[:7] = xyz_wxyz
                eef_configuration.update(original_q)
            if robot_configuration is not None and qs is not None:
                if i == len(qs):
                    break
                robot_configuration.update(qs[i])
                i += 1
            vis.sync()
            time.sleep(rate)
        vis.close()


def optimize_robot_configuration(
    robot_configuration: IndexedConfiguration,
    eef_configuration: Optional[IndexedConfiguration] = None,
    eef_target: Optional[np.ndarray] = None,
    eef_target_type: Literal["pose", "wxyz_xyz"] = "pose",
    eef_frame_name: str = "gripper0_right_right_gripper",
    eef_frame_type: str = "body",
    update_gripper_qpos: bool = False,
    global_opt_max_iter: int = 200,
    global_opt_trials: int = 6,
    start_q: Optional[np.ndarray] = None,  # used by sequential IK solvers
    start_q_weights: Optional[np.ndarray] = None,
    retract_q: Optional[np.ndarray] = None,
    retract_q_weights: Optional[np.ndarray] = None,
    optimization_type: Literal["scipy", "eef_interp_mink"] = "scipy",
    eef_interp_mink_mp: Optional[EEFInterpMinkMotionPlanner] = None,
    verbose: bool = False,
) -> Optional[np.ndarray]:
    """
    Optimize the robot configuration to reach the desired end-effector pose.

    Require either eef_configuration or eef_target.

    TODO: to enable better joint-joint motion planning, should find a robot q that is close to start_q.
    """
    target_wxyz_xyz = None
    if eef_configuration is not None:
        target_wxyz_xyz = np.concatenate(
            [
                eef_configuration.q[eef_configuration.robot_idxs][3:7],
                eef_configuration.q[eef_configuration.robot_idxs][:3],
            ]
        )
    elif eef_target is not None:
        if eef_target_type == "pose":
            target_wxyz_xyz = np.concatenate(
                [mat2quat(eef_target[:3, :3]), eef_target[:3, 3]]
            )
        elif eef_target_type == "wxyz_xyz":
            target_wxyz_xyz = eef_target
    else:
        assert (
            target_wxyz_xyz is not None
        ), "Either eef_configuration or eef_target is required."

    original_q = robot_configuration.q[robot_configuration.robot_idxs].copy()
    if optimization_type == "scipy":
        # Define the objective function
        def objective(q):
            robot_configuration.update(q)
            current_wxyz_xyz = robot_configuration.get_transform_frame_to_world(
                eef_frame_name, eef_frame_type
            ).wxyz_xyz
            if retract_q is not None:
                return np.linalg.norm(
                    current_wxyz_xyz - target_wxyz_xyz
                ) + 0.002 * np.linalg.norm((q - retract_q) * retract_q_weights)
            return np.linalg.norm(current_wxyz_xyz - target_wxyz_xyz)

        # TODO: update this for getting bounds of the relevant joints!
        # Get joint limits
        qmin = robot_configuration.model.jnt_range[:, 0].copy()
        qmax = robot_configuration.model.jnt_range[:, 1].copy()
        qmin = qmin[robot_configuration.robot_idxs]
        qmax = qmax[robot_configuration.robot_idxs]
        bounds = [(qmin[i], qmax[i]) for i in range(len(qmin))]

        # Initial guess
        initial_pose = robot_configuration.q[robot_configuration.robot_idxs]

        # Perform optimization
        result = minimize(objective, initial_pose, bounds=bounds)
        robot_configuration.update(result.x)
        current_wxyz_xyz = robot_configuration.get_transform_frame_to_world(
            eef_frame_name, eef_frame_type
        ).wxyz_xyz
        pos_diff = np.linalg.norm(target_wxyz_xyz[4:] - current_wxyz_xyz[4:])
        quat_diff = np.linalg.norm(target_wxyz_xyz[:4] - current_wxyz_xyz[:4])
        success = False
        if pos_diff < 1e-3 and quat_diff < 1e-3:
            if verbose:
                logging.info(f"Success! pos_diff: {pos_diff} quat_diff: {quat_diff}")
                if retract_q is not None:
                    logging.info(
                        f"retract diff: {np.linalg.norm((result.x - retract_q) * retract_q_weights)}"
                    )
                logging.info(f"Minimize res {result.fun}")
            success = True  # false for now b/c need to lead to failed grasping;
            # TODO: maybe skip dual annealing if already optimized for constraint

        if not success and result.fun > 1e-3:
            if verbose:
                logging.info("Optimization failed to converge.")
                logging.info("opt res: ", result.fun)

            # Initialize variables to track the best solution
            best_result = None
            min_retract_diff = float("inf")
            for _ in range(global_opt_trials):
                initial_pose = np.random.uniform(qmin, qmax)
                result = dual_annealing(
                    objective,
                    bounds=bounds,
                    x0=initial_pose,
                    maxiter=global_opt_max_iter,
                    maxfun=1e4,
                )

                # Update the robot configuration with the result
                robot_configuration.update(result.x)
                current_wxyz_xyz = robot_configuration.get_transform_frame_to_world(
                    eef_frame_name, eef_frame_type
                ).wxyz_xyz
                pos_diff = np.linalg.norm(target_wxyz_xyz[4:] - current_wxyz_xyz[4:])
                quat_diff = np.linalg.norm(target_wxyz_xyz[:4] - current_wxyz_xyz[:4])
                success = False

                if pos_diff < 1e-3 and quat_diff < 1e-3:
                    # Calculate retract difference if retract_q is given
                    if retract_q is not None:
                        retract_diff = np.linalg.norm(
                            (result.x - retract_q) * retract_q_weights
                        )
                        if verbose:
                            logging.info(
                                f"Success! pos_diff: {pos_diff} quat_diff: {quat_diff}"
                            )
                            logging.info(
                                f"target_xyz: {target_wxyz_xyz[4:]}; current_xyz: {current_wxyz_xyz[4:]}"
                            )
                            logging.info(f"retract diff: {retract_diff}")
                            logging.info(f"retract_q: {retract_q}")
                            logging.info(f"result.x: {result.x}")
                            logging.info(
                                f"iter {_} new opt res in the loop: ", result.fun
                            )

                        # Track the best result based on retract difference
                        if retract_diff < min_retract_diff:
                            min_retract_diff = retract_diff
                            best_result = result
                            if verbose:
                                logging.info(
                                    f"Found a better result with retract diff: {retract_diff}"
                                )
                    if verbose:
                        logging.info(
                            f"Failed to reach the target pose after {global_opt_trials} trials."
                            f"pos_diff: {pos_diff} quat_diff: {quat_diff}"
                        )

            if best_result is None:
                if verbose:
                    logging.info(f"target_wxyz_xyz: {target_wxyz_xyz}")
                    eef_pose = make_pose(
                        target_wxyz_xyz[4:],
                        quat2mat(np.roll(target_wxyz_xyz[:4], -1)),
                    )
                    create_passive_viewer(
                        eef_configuration.model,
                        eef_configuration.data,
                        eef_configuration=eef_configuration,
                        eef_pose=eef_pose,
                        rate=5,
                    )
                    robot_configuration.update(result.x)
                    logging.info(f"viewing solution of trial {_}")
                    create_passive_viewer(
                        robot_configuration.model,
                        robot_configuration.data,
                        qs=[initial_pose],
                        robot_configuration=robot_configuration,
                    )

            result = best_result

        if result is None:
            robot_configuration.update(original_q)
            return None

        if not np.all(result.x >= qmin) and np.all(result.x <= qmax):
            logging.info("Result is out of bounds")
            logging.info(f"result: {result.x}")
            logging.info(f"qmin: {qmin}")
            logging.info(f"qmax: {qmax}")
            import ipdb

            ipdb.set_trace()
            # Ensure the optimized result is within joint limits
            result.x = np.clip(result.x, qmin, qmax)
        out_q = result.x

    elif optimization_type == "curobo":
        raise NotImplementedError("Curobo optimization is not implemented yet.")
    elif optimization_type == "eef_interp_mink":
        X_start = robot_configuration.get_transform_frame_to_world(
            eef_frame_name, eef_frame_type
        ).as_matrix()
        X_goal = make_pose(
            target_wxyz_xyz[4:], quat2mat(np.roll(target_wxyz_xyz[:4], -1))
        )
        robot_qs, eef_wxyz_xyzs = eef_interp_mink_mp.plan(
            X_start,
            X_goal,
            start_q[:7],
            robot_configuration=robot_configuration,
            q_retract=retract_q,
            retract_q_weights=retract_q_weights,
            eef_frame_name=eef_frame_name,
            eef_frame_type=eef_frame_type,
        )
        out_q = robot_qs[-1]
        if verbose:
            create_passive_viewer(
                robot_configuration.model,
                robot_configuration.data,
                qs=robot_qs,
                robot_configuration=robot_configuration,
            )

    if not update_gripper_qpos:
        assert len(out_q) == 9, "Expected 9 joint values for the robot."
        out_q[-2:] = original_q[-2:]
    # Return the optimized pose
    return out_q


def generate_lift_actions(
    last_action: List[float], retract_height: float = 0.04, retract_steps: int = 5
) -> List[List[float]]:
    """
    Generate a list of retracting actions from the last action.

    Args:
        last_action: last original action before the retract actions.
        retract_height (float): Total height to retract along the z-axis. Default is 0.04.
        retract_steps (int): Number of steps for the retract motion. Default is 4.
    """
    from scipy.spatial.transform import Rotation

    new_actions = []
    z_increment = [retract_height / retract_steps for _ in range(retract_steps)]
    for i in range(retract_steps):
        if i == 0:
            new_action = last_action.copy()
        else:
            new_action = new_actions[-1].copy()

        pos = np.array(new_action[:3], dtype=float)
        quat_xyzw = axisangle2quat(new_action[3:6])
        # Create a rotation object from the quaternion (expects [x, y, z, w] format for scipy)
        rotation = Rotation.from_quat(quat_xyzw)
        # Define the z-offset vector in the local frame
        z_offset_vector = np.array([0, 0, -z_increment[i]])
        # Rotate the z-offset vector to align with the current orientation
        z_offset_vector_rotated = rotation.apply(z_offset_vector)
        # Update the position by adding the rotated z-offset vector
        new_pos = pos + z_offset_vector_rotated
        new_action[:3] = new_pos
        new_actions.append(new_action)
    return new_actions


def generate_open_gripper_actions(
    last_action: List[float], open_gripper_steps: int = 10
) -> List[List[float]]:
    new_actions = []
    for _ in range(open_gripper_steps):
        new_action = last_action.copy()
        new_action[6] = -1
        new_actions.append(new_action)
    return new_actions


def generate_close_gripper_actions(
    last_action: List[float], open_gripper_steps: int = 10
) -> List[List[float]]:
    new_actions = []
    for _ in range(open_gripper_steps):
        new_action = last_action.copy()
        new_action[6] = 1
        new_actions.append(new_action)
    return new_actions


def get_scale_factor(
    demo_obj_geoms_size: Dict[str, np.ndarray],
    current_obj_geoms_size: Dict[str, np.ndarray],
) -> np.ndarray:
    # Find the first key in both dicts that matches and has all non-zero components
    matching_key = next(
        (
            key
            for key in demo_obj_geoms_size
            if key in current_obj_geoms_size
            and np.all(demo_obj_geoms_size[key] > 0)
            and np.all(current_obj_geoms_size[key] > 0)
        ),
        None,
    )
    if matching_key is None:
        # Fall back to first matching key even with zeros
        matching_key = next(
            (key for key in demo_obj_geoms_size if key in current_obj_geoms_size), None
        )
    if matching_key is None:
        raise ValueError(
            "No matching key found in the demo's geometries and current obs' geometries. Cannot compute scale factor. "
            f"Likely cause: obj geometry naming mismatch between envs from which obj geoms were obtained.\n"
            f"Geoms from demo env: {demo_obj_geoms_size}.\n"
            f"Geoms from current env's obs: {current_obj_geoms_size}"
        )

    demo_obj_geom_size = np.array(demo_obj_geoms_size[matching_key])
    current_obj_geom_size = np.array(current_obj_geoms_size[matching_key])
    # Ensure both arrays have zeros in the same indices
    diff_zero_indices = np.logical_xor(
        demo_obj_geom_size == 0, current_obj_geom_size == 0
    )
    assert not diff_zero_indices.any(), (
        "Demo and current object geometries must have zeros in the same indices. "
        f"Demo geoms: {demo_obj_geoms_size}, Current geoms: {current_obj_geom_size}"
    )

    demo_obj_geom_size[np.where(demo_obj_geom_size == 0)] = 1
    current_obj_geom_size[np.where(current_obj_geom_size == 0)] = 1

    # Calculate the scale factor difference
    scale_factor = current_obj_geom_size / demo_obj_geom_size
    return scale_factor


def optimize_robot_configuration_kp(
    configuration: IndexedConfiguration,
    target_keypoints: np.ndarray,
    initial_pose: np.ndarray,
    selected_indices: Optional[np.ndarray] = None,
    verbose: bool = False,
    max_iter: int = 200,
    global_opt: bool = True,
    local_opt: bool = True,
    constraint_gripper_symmetry: bool = True,
    qmin: Optional[np.ndarray] = None,
    qmax: Optional[np.ndarray] = None,
) -> np.ndarray:
    """
    Optimize the (robot's) configuration to minimize keypoint distance while keeping close to the initial pose.

    Args:
        configuration: Configuration object for the robot.
        target_keypoints: Desired keypoint positions.
        initial_pose: Initial joint pose.
        selected_indices: Indices of joints to optimize (others remain fixed).
        verbose: If True, logs optimization details.
        max_iter: Maximum iterations for optimization.
        global_opt: If True, perform global optimization.
        local_opt: If True, perform local refinement.
        constraint_gripper_symmetry: If True, enforces gripper symmetry.
        qmin: Lower joint limits.
        qmax: Upper joint limits.

    Returns:
        Optimized joint configuration.
    """

    # If no indices are specified, optimize all joints
    if selected_indices is None:
        selected_indices = np.arange(len(initial_pose))

    # Extract initial values for the selected joints
    x0 = initial_pose[selected_indices]

    # Define the objective function
    def objective(x):
        full_x = initial_pose.copy()
        full_x[selected_indices] = x
        configuration.update(full_x)

        try:
            curr_keypoints = configuration.get_keypoints()
        except Exception as e:
            logging.error(f"Error in getting keypoints: {e}")
            return 1e6

        distance = np.linalg.norm(target_keypoints - curr_keypoints)

        # Keep close to the initial pose
        distance += 0.005 * np.linalg.norm(x - x0)
        if constraint_gripper_symmetry:
            distance += 0.1 * np.abs(
                np.sum(configuration.q[configuration.robot_idxs][-2:])
            )

        return distance

    if qmin is None:
        qmin = configuration.model.jnt_range[:, 0].copy()[configuration.robot_idxs]
    if qmax is None:
        qmax = configuration.model.jnt_range[:, 1].copy()[configuration.robot_idxs]

    qmin_selected = qmin[selected_indices]
    qmax_selected = qmax[selected_indices]
    bounds = list(zip(qmin_selected, qmax_selected))

    def callback_global(x, f, context):
        return f < 5e-4  # Terminate early if below threshold

    def callback_local(intermediate_result):
        if intermediate_result.fun < 5e-4:
            raise StopIteration("Terminating optimization as threshold reached.")

    # Step 1: Global optimization using dual_annealing
    if global_opt:
        result = dual_annealing(
            objective,
            bounds=bounds,
            x0=x0,
            maxiter=max_iter,
            maxfun=1e4,
            callback=callback_global,
        )
        x0 = result.x
        if verbose:
            logging.info(
                f"Global optimization cost {result.fun:.3f}, niter {result.nit}"
            )

    # Step 2: Local refinement using minimize
    if local_opt:
        result = minimize(objective, x0, bounds=bounds, callback=callback_local)
        if verbose:
            logging.info(f"Local refinement cost {result.fun:.3f}, niter {result.nit}")

    # Update final pose
    final_pose = initial_pose.copy()
    final_pose[selected_indices] = result.x

    return final_pose


def compute_X_W_eef_des(
    constraint: Constraint,
    obs: Dict,
    robot_configuration: IndexedConfiguration,
    eef_configuration: IndexedConfiguration,
    X_obj0_transf: Optional[np.ndarray] = None,
    X_obj1_transf: Optional[np.ndarray] = None,
    X_eef_transf: Optional[np.ndarray] = None,
    src_t_idx: int = 0,
    eef_body_frame_name: str = "gripper0_right_right_gripper",
    eef_site_frame_name: str = "gripper0_right_grip_site",
    ignore_scaling: bool = False,
) -> np.ndarray:
    """
    Compute the desired end-effector pose based on the constraint and observation data.

    Args:
        constraint: Constraint object.
        obs: Observation dictionary.
        robot_configuration: Robot configuration object.
        eef_configuration: End-effector configuration object.
        X_obj0_transf: Transformation matrix for object 0.
        X_obj1_transf: Transformation matrix for object 1.
        X_eef_transf: Transformation matrix for the end-effector.
        src_t_idx: Source time index.
        eef_body_frame_name: End-effector body frame name.
        eef_site_frame_name: End-effector site frame name.
        ignore_scaling: If True, ignores scaling factor. Used for debugging
            when we're only using the source objects sizes, such as in the non-X-wide envs.
    """
    if X_obj0_transf is None:
        X_obj0_transf = np.eye(4)
    if X_obj1_transf is None:
        X_obj1_transf = np.eye(4)
    if X_eef_transf is None:
        X_eef_transf = np.eye(4)
    eef_configuration.update(
        # pos is from the site --- we want pos to be for the body
        np.concatenate(
            [
                obs["robot0_eef_pos_body"],
                obs["robot0_eef_quat_site"][np.array([3, 0, 1, 2])],
                obs["robot0_gripper_qpos"],
            ]
        )
    )

    # Set the mocap positions to keypoints in the world frame
    def set_mocap_pos_and_update_viewer(env: CPEnv, mocap_pos: np.ndarray):
        max_mocaps = min(env.env.env.sim.data.mocap_pos.shape[0], len(mocap_pos))
        env.env.env.sim.data.mocap_pos[:max_mocaps] = mocap_pos[:max_mocaps]
        env.env.env.sim.forward()

    if constraint.constraint_type == "robot-object":
        # Extract necessary data for robot-object constraints
        task_relev_obj = constraint.obj_names[0]
        X_W_obj_demo = (
            constraint.src_obj_pose[task_relev_obj][src_t_idx] @ X_obj0_transf
        )
        # Get the scale factor of the demo object and the current object
        demo_obj_geoms_size: Dict[str, np.ndarray] = constraint.src_obj_geoms_size[
            task_relev_obj
        ][src_t_idx]
        current_obj_geoms_size: Dict[str, np.ndarray] = obs[
            f"{task_relev_obj}_geoms_size"
        ]

        # Use the helper function to get the scale factor
        scale_factor = (
            get_scale_factor(demo_obj_geoms_size, current_obj_geoms_size)
            if not ignore_scaling
            else 1.0
        )

        # Scale the original keypoints to the current object's scale
        scaled_keypoints = (
            constraint.keypoints_obj_frame[task_relev_obj][src_t_idx] * scale_factor
        )

        # Get the object's pose in the world frame
        X_W_obj_curr = obs[f"{task_relev_obj}_pose"] @ np.linalg.inv(X_obj0_transf)
        # Transform the keypoints to the world frame
        keypoints_world_frame = transform_keypoints(scaled_keypoints, X_W_obj_curr)
        link_to_kpts = {
            link_name: keypoints[src_t_idx]
            for link_name, keypoints in constraint.keypoints_robot_link_frame.items()
        }
        X_W_eefsite_demo = np.eye(4)
        X_W_eefsite_demo[:3, 3] = constraint.src_obs["robot0_eef_pos"][src_t_idx]
        X_W_eefsite_demo[:3, :3] = quat2mat(
            constraint.src_obs["robot0_eef_quat_site"][src_t_idx]
        )
        # yeah, might be something wrong here, esp if X_W_obj0 is e.g. a rotation
        # X_W_eefsite_demo @= X_eef_transf  # doing this isn't enough to update the keypoints --- need to update original keypoints
        # on the task relevant objects too ... however those were created in the complete_constraint part ...
        X_obj_eefsite_demo = np.linalg.inv(X_W_obj_demo) @ X_W_eefsite_demo

        # Get current object pose in the world frame
        X_W_obj_curr = obs[f"{task_relev_obj}_pose"]
        X_W_eefsite_des = X_W_obj_curr @ X_obj_eefsite_demo

        # Compute the mapping X_eef_eefsite
        # TODO: can we simplify the need for having both hardcoded names e.g. gripper0_right_right_gripper
        X_W_eef_curr = robot_configuration.get_transform_frame_to_world(
            eef_body_frame_name, "body"
        ).as_matrix()
        X_W_eefsite_curr = robot_configuration.get_transform_frame_to_world(
            eef_site_frame_name, "site"
        ).as_matrix()
        X_eefsite_eef = np.linalg.inv(X_W_eefsite_curr) @ X_W_eef_curr
        X_W_eef_des = X_W_eefsite_des @ X_eefsite_eef
        X_W_eef_des = (
            X_W_eef_des @ X_eef_transf
        )  # it is clear that this is incorrect then ...

        assert (
            np.allclose(X_eef_transf, np.eye(4))
            or np.allclose(X_eef_transf[:3, :3], R.from_euler("z", np.pi).as_matrix())
        ), "X_eef_transf must be either np.eye(4) or a rotation of 180 degrees about the x-axis"

        flip_eef = not np.allclose(X_eef_transf, np.eye(4), rtol=1e-5, atol=1e-8)
        robot_configuration.set_keypoints(link_to_kpts, flip_eef=flip_eef)
        eef_configuration.set_keypoints(link_to_kpts, flip_eef=flip_eef)
        initial_q = xyz_wxyz_gripper = np.concatenate(
            [
                X_W_eef_des[:3, 3],
                mat2quat(X_W_eef_des[:3, :3]),
                eef_configuration.get_robot_qpos()[-2:],
            ]
        )

        eef_configuration.update(xyz_wxyz_gripper)

        # Optimize the robot configuration so its keypoints align with the target.
        initial_q: np.ndarray = eef_configuration.get_robot_qpos().copy()
        initial_q = optimized_q = optimize_robot_configuration_kp(
            configuration=eef_configuration,
            target_keypoints=keypoints_world_frame,
            initial_pose=initial_q,
            max_iter=500,
            verbose=True,
            global_opt=False,
            qmin=np.concatenate(
                [-3 * np.ones(7), eef_configuration.model.jnt_range[-2:, 0]]
            ),
            qmax=np.concatenate(
                [3 * np.ones(7), eef_configuration.model.jnt_range[-2:, 1]]
            ),
            selected_indices=np.array([0, 1, 2, 3, 4, 5, 6, 7, 8]),
        )
        eef_configuration.update(optimized_q)
        X_W_eef_des = eef_configuration.get_transform_frame_to_world(
            eef_body_frame_name, "body"
        ).as_matrix()
        # think we don't want the @ X_eef_transf if using keypoints who've taken into account the flip
    elif constraint.constraint_type == "object-object":
        if constraint.constraint_type == "object-object":
            # only keep keypoints attached to the kinematically 'actuated' object
            link_to_kpts = {
                link_name: keypoints[src_t_idx]
                for link_name, keypoints in constraint.keypoints_obj_frame.items()
                if link_name in constraint.obj_to_parent_attachment_frame
                and constraint.obj_to_parent_attachment_frame[link_name] is not None
            }

            task_relev_obj_0 = constraint.obj_names[0]
            task_relev_obj_1 = constraint.obj_names[1]

            # Get the scale factor of the demo object and the current object
            demo_obj_geoms_size_0: Dict[str, np.ndarray] = (
                constraint.src_obj_geoms_size[task_relev_obj_0][src_t_idx]
            )
            current_obj_geoms_size_0: Dict[str, np.ndarray] = obs[
                f"{task_relev_obj_0}_geoms_size"
            ]

            demo_obj_geoms_size_1: Dict[str, np.ndarray] = (
                constraint.src_obj_geoms_size[task_relev_obj_1][src_t_idx]
            )
            current_obj_geoms_size_1: Dict[str, np.ndarray] = obs[
                f"{task_relev_obj_1}_geoms_size"
            ]

            # Use the helper function to get the scale factor
            scale_factor_0 = (
                get_scale_factor(demo_obj_geoms_size_0, current_obj_geoms_size_0)
                if not ignore_scaling
                else np.ones(3)
            )
            scale_factor_1 = (
                get_scale_factor(demo_obj_geoms_size_1, current_obj_geoms_size_1)
                if not ignore_scaling
                else np.ones(3)
            )

            # Scale the original keypoints to the current object's scale
            pts_0_homog = np.hstack(
                [
                    constraint.keypoints_obj_frame[task_relev_obj_0][src_t_idx],
                    np.ones(
                        (
                            constraint.keypoints_obj_frame[task_relev_obj_0][
                                src_t_idx
                            ].shape[0],
                            1,
                        )
                    ),
                ]
            )
            P_obj_target_0 = (X_obj0_transf @ pts_0_homog.T).T[
                ..., :3
            ]  # apply the obj0 transforms
            # transforms mightn't be fully correct atm ...
            scaled_keypoints_0 = P_obj_target_0 * scale_factor_0
            link_to_kpts[task_relev_obj_0] = scaled_keypoints_0

            pts_1_homog = np.hstack(
                [
                    constraint.keypoints_obj_frame[task_relev_obj_1][src_t_idx],
                    np.ones(
                        (
                            constraint.keypoints_obj_frame[task_relev_obj_1][
                                src_t_idx
                            ].shape[0],
                            1,
                        )
                    ),
                ]
            )
            P_obj_target_1 = (X_obj1_transf @ pts_1_homog.T).T[..., :3]
            # get object keypoints in world frame: will be used in optimization q* = argmin_q ||P_W_goal - P_W_robot_kp(q)||^2
            scaled_keypoints_1 = P_obj_target_1 * scale_factor_1

            P_W_target = transform_keypoints(
                scaled_keypoints_1,
                obs[f"{task_relev_obj_1}_pose"] @ np.linalg.inv(X_obj1_transf),
            )

            use_weld_heuristic_action: bool = True
            if use_weld_heuristic_action:
                constraint.src_gripper_action[src_t_idx] = (
                    1  # hack for now to test if we needed to match actions rather than states
                )

        # Extract necessary data for object-object constraints
        X_W_obj0_demo = (
            constraint.src_obj_pose[task_relev_obj_0][src_t_idx] @ X_obj0_transf
        )
        X_W_obj1_demo = (
            constraint.src_obj_pose[task_relev_obj_1][src_t_idx] @ X_obj1_transf
        )

        X_obj0_obj1_demo = np.linalg.inv(X_W_obj0_demo) @ X_W_obj1_demo

        # Compute desired object 0 pose
        X_W_obj1_curr = obs[f"{task_relev_obj_1}_pose"] @ np.linalg.inv(X_obj1_transf)
        X_W_obj0_des = X_W_obj1_curr @ np.linalg.inv(X_obj0_obj1_demo)

        # Compute desired end-effector pose
        X_W_eef = robot_configuration.get_transform_frame_to_world(
            eef_body_frame_name, "body"
        ).as_matrix()
        X_W_obj0 = robot_configuration.get_transform_frame_to_world(
            constraint.obj_names[0], "body"
        ).as_matrix()
        X_eef_obj0 = np.linalg.inv(X_W_eef) @ X_W_obj0
        X_W_eef_des = X_W_obj0_des @ np.linalg.inv(X_eef_obj0)

        flip_eef = not np.allclose(X_eef_transf, np.eye(4), rtol=1e-5, atol=1e-8)
        robot_configuration.set_keypoints(link_to_kpts, flip_eef=flip_eef)
        eef_configuration.set_keypoints(link_to_kpts, flip_eef=flip_eef)

        initial_q = xyz_wxyz_gripper = np.concatenate(
            [
                X_W_eef_des[:3, 3],
                mat2quat(X_W_eef_des[:3, :3]),
                eef_configuration.get_robot_qpos()[-2:],
            ]
        )

        eef_configuration.update(xyz_wxyz_gripper)

        # Optimize the robot configuration so its keypoints align with the target.
        initial_q: np.ndarray = eef_configuration.get_robot_qpos().copy()
        for _ in range(2):
            initial_q = optimized_q = optimize_robot_configuration_kp(
                configuration=eef_configuration,
                target_keypoints=P_W_target,
                initial_pose=initial_q,
                max_iter=500,
                verbose=True,
                global_opt=False,
                qmin=np.concatenate(
                    [-3 * np.ones(7), eef_configuration.model.jnt_range[-2:, 0]]
                ),
                qmax=np.concatenate(
                    [3 * np.ones(7), eef_configuration.model.jnt_range[-2:, 1]]
                ),
                selected_indices=np.array([0, 1, 2, 3, 4, 5, 6, 7, 8])
                if _ == 0
                else np.arange(9),
                # pos only first, then pos + quat together
            )
            eef_configuration.update(optimized_q)

        if obs.get("env", None) is not None:
            kp_curr = robot_configuration.get_keypoints()
            set_mocap_pos_and_update_viewer(obs["env"], kp_curr)
        X_W_eef_des = eef_configuration.get_transform_frame_to_world(
            eef_body_frame_name, "body"
        ).as_matrix()

    return X_W_eef_des


def rotation_about_point(
    rot_axis: str, rot_param: float, origin: np.ndarray
) -> np.ndarray:
    """
    Construct a 4x4 SE(3) matrix representing a rotation about a point in space.

    Parameters:
        rot_axis (str): Axis of rotation ('x', 'y', or 'z').
        rot_param (float): Rotation angle (in radians).
        origin (np.ndarray): 3D point (shape (3,)) about which to rotate.

    Returns:
        np.ndarray: 4x4 SE(3) transformation matrix.
    """
    # Build rotation matrix
    # Build translation matrices
    T_neg_o = np.eye(4)
    T_neg_o[:3, 3] = -origin

    Rm = np.eye(4)
    Rm[:3, :3] = R.from_euler(rot_axis, rot_param).as_matrix()

    T_o = np.eye(4)
    T_o[:3, 3] = origin

    # Rotate about the point: T_o * R * T_-o
    return T_o @ Rm @ T_neg_o


def generate_se3_transforms(transforms: list[dict]) -> list[np.ndarray]:
    """
    Generate a list of SE(3) transformation matrices from a list of transforms.

    Parameters:
        transforms (list[dict]): List of transformation specifications, each containing:
                                 - 'rot_axis' (str): Rotation axis ('x', 'y', 'z').
                                 - 'rot_param' (float): Rotation angle.
                                 - 'translation' (np.ndarray): 3D translation vector.

    Returns:
        list[np.ndarray]: List of 4x4 SE(3) transformation matrices.
    """
    result = []
    for transform_params in transforms:
        if "rot_axis" in transform_params and "rot_param" in transform_params:
            rot = R.from_euler(
                transform_params["rot_axis"], transform_params["rot_param"]
            ).as_matrix()
            if "rotation_origin" in transform_params:
                # import ipdb; ipdb.set_trace()
                T_rot = rotation_about_point(
                    transform_params["rot_axis"],
                    transform_params["rot_param"],
                    np.array(transform_params["rotation_origin"]),
                )
            else:
                T_rot = np.eye(4)
                T_rot[:3, :3] = rot
        else:
            T_rot = np.eye(4)
        T_final = T_rot
        # Apply translation if any
        if "translation" in transform_params:
            T_trans = np.eye(4)
            T_trans[:3, 3] = transform_params["translation"]
            # Apply order: translation THEN rotation (local), or rotation THEN translation (world)
            T_final = T_trans @ T_final  # you can change order if needed
        result.append(T_final)  # this would be used as `pose @ T_final`
    return result


def find_best_transform(
    constraint: Constraint, obs: Dict, robot_configuration, eef_configuration
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Find best object-centric constraint-satisfying symmetry transforms.
    利用物体或夹爪的对称性，寻找一个最“顺手”、最容易达到的抓取（或操作）姿态。
    Objective: argmin_{Xobj1_Xobj2_Xee} dist(X_ee_des(Xobj1_Xobj2_Xee), X_ee_curr).

    Extensions: check if Xobj1_Xobj2_Xee is trackable.
    """
    # Adjust weights: 1 m translation is 5 units per meter (i.e. 0.01m -> 0.05) and 1 per radian rotation.
    w_trans: float = 5.0  # Weight for translation error (per meter)
    w_rot: float = 1.0  # Weight for rotation error (per radian)

    best_score: float = float("inf")
    best_X_obj0_transf = np.eye(4)
    best_X_obj1_transf = np.eye(4)
    best_X_eef_transf = np.eye(4)

    X_obj0_transf_candidates = [np.eye(4)]
    X_obj1_transf_candidates = [np.eye(4)]
    X_eef_transf_candidates = [np.eye(4)]

    curr_eef_pos = obs["robot0_eef_pos"]
    # curr_eef_quat_xyzw = obs["robot0_eef_quat"]
    curr_eef_quat_xyzw_site = obs[
        "robot0_eef_quat_site"
    ]  # use site b/c body doesn't exist in eef_configuration

    for sym in constraint.symmetries:
        if sym.type == "obj0":
            if sym.skip_default:
                X_obj0_transf_candidates = []
            X_obj0_transf_candidates.extend(generate_se3_transforms(sym.transforms))
        elif sym.type == "obj1":
            X_obj1_transf_candidates.extend(generate_se3_transforms(sym.transforms))
        elif sym.type == "eef":
            X_eef_transf_candidates.extend(generate_se3_transforms(sym.transforms))

    # hardcoded for parallel jaw gripper
    if constraint.constraint_type == "robot-object" and constraint.reflect_eef:
        rot_z_180_transf = np.eye(4)
        rot_z_180_transf[:3, :3] = R.from_euler("z", np.pi).as_matrix()
        X_eef_transf_candidates.append(rot_z_180_transf)

    for X_eef_transf in X_eef_transf_candidates:
        for X_obj0_transf in X_obj0_transf_candidates:
            for X_obj1_transf in X_obj1_transf_candidates:
                # Update the configuration; this call updates eef_configuration.
                _ = compute_X_W_eef_des(
                    constraint,
                    obs,
                    robot_configuration,
                    eef_configuration,
                    X_obj0_transf=X_obj0_transf,
                    X_obj1_transf=X_obj1_transf,
                    X_eef_transf=X_eef_transf,
                )
                qpos = eef_configuration.get_robot_qpos()
                des_eef_pos = qpos[0:3]
                # Calculate translation error (Euclidean distance).
                trans_error: float = np.linalg.norm(des_eef_pos - curr_eef_pos)
                des_eef_quat_wxyz_site = (
                    eef_configuration.get_transform_frame_to_world(
                        "gripper0_right_grip_site", "site"
                    )
                    .rotation()
                    .wxyz
                )
                des_eef_quat_xyzw_site = des_eef_quat_wxyz_site[[1, 2, 3, 0]]
                quat_diff_site = R.from_quat(
                    curr_eef_quat_xyzw_site
                ).inv() * R.from_quat(des_eef_quat_xyzw_site)
                rot_error_site: float = quat_diff_site.magnitude()
                # Compute combined weighted score.
                score = w_trans * trans_error + w_rot * rot_error_site

                if score < best_score:
                    best_score = score
                    best_X_obj0_transf = X_obj0_transf
                    best_X_obj1_transf = X_obj1_transf
                    best_X_eef_transf = X_eef_transf

    return best_X_obj0_transf, best_X_obj1_transf, best_X_eef_transf


class CPPolicy:
    """核心职责是根据当前的观测，结合预定义的约束序列，计算出机器人下一步该怎么动。"""
    def __init__(
        self,
        constraint_sequences: List[List[Constraint]],  # demos的constraint集合
        motion_planner: MotionPlanner,
        robot_configuration: IndexedConfiguration,
        eef_configuration: Optional[IndexedConfiguration] = None,
        eef_interp_mink_motion_planer: Optional[
            EEFInterpMinkMotionPlanner
        ] = None,  # for IK solving via mink + interp
        viz_robot_kpts: bool = False,
        env: CPEnv = None,
        verbose: bool = False,
        original_xml: str = None,
        ignore_scaling: bool = False,
    ):
        self.motion_planner = motion_planner  # 保存运动规划器。这是用于计算从点 A 到点 B 无碰撞路径的核心工具
        self.eef_interp_mink_motion_planer = eef_interp_mink_motion_planer  # 保存辅助规划器（Mink）。这是一个特定的基于 Mink 库的规划器，通常用于更精细的逆运动学（IK）求解或简单的插值运动。
        self.constraint_sequences = constraint_sequences
        # 保存机器人全身构型对象
        self.robot_configuration = robot_configuration  # used for kinematics queries in e.g. constraint solving
        # 保存末端执行器（EEF）构型对象。
        self.eef_configuration = (
            eef_configuration  # used for kinematics queries in e.g. constraint solving
        )
        self.ignore_scaling = ignore_scaling  # “忽略缩放”开关。如果演示中的物体大小和新环境中的物体大小不一致，通常需要缩放关键点。
        self.viz_robot_kpts = viz_robot_kpts  # “可视化关键点”开关。
        self.current_constraint_idx = 0  # 当前选用了第几套demo。
        self.current_constraint_sequence_idx = 0  # 当前执行到了demo里的第几步（比如是第0步“靠近”，还是第1步“抓取”）。
        self.action_queue: List[
            np.ndarray
        ] = []  # Stores multiple actions to be executed
        # False:我现在很忙，要么在规划路径，要么在执行动作，别打扰我。
        #   发生时机：1）刚初始化时。2）正在规划出了一长串动作序列（action_queue 里有货）。3）正在一步步执行这些动作的过程中。
        # True:我刚才规划的一整套动作做完了，快来看看我成功了没？如果成功了，我要准备进入下一关了。
        #   发生时机：当 action_queue（动作队列）变空了（即 pop 完了最后一个动作）。
        self.check_constraint = (
            False  # Indicates when to check current constraint is satisfied
        )
        self.curr_motion_plan_steps: int = -1  # 记录当前阶段规划了多少步用于“移动”（Motion）
        self.curr_constraint_steps: int = -1  # 记录多少步用于“执行约束”（Constraint）
        self.max_iter: int = 100  # 逆运动学（IK）求解器的最大迭代次数。
        self.verbose: bool = verbose
        self.env = env  # used to update robot configuration for attachments
        self.original_xml = original_xml  # b/c get_xml() is overridden  # 保存原始 XML 模型字符串

    def reset(self):
        """Reset the policy to the first constraint."""
        self.current_constraint_idx = 0
        self.check_constraint = False
        self.action_queue = []

    @property
    def done(self) -> bool:
        """Check if all constraints in current constraint sequence have been attempted to be solved."""
        # (not self.action_queue)：手头的动作做完了吗？
        # idx >= len(sequence)：demo到头了吗？
        return (not self.action_queue) and self.current_constraint_idx >= len(
            self.current_constraint_sequence
        )

    @property
    def current_constraint_sequence(self) -> List[Constraint]:
        """Get the sequence of constraints."""
        # 当前这集demo
        return self.constraint_sequences[self.current_constraint_sequence_idx]

    @property
    def current_constraint(self) -> Constraint:
        """Get the current constraint to be solved."""
        # 当前这集demo的当前constraint
        return self.current_constraint_sequence[self.current_constraint_idx]

    def update_constraint(self, env: CPEnv, constraint: Constraint):
        """
        Constraint 对象本身只存储了 “过去（Demo）” 的数据，而 CPPolicy 在运行时需要知道 “现在（Env）” 的情况。这个函数负责把环境里的实时数据抓取出来，塞进 Constraint 对象里，供后续计算使用。
        Update constraint information with poses and transforms from the environment.
        """
        for task_relevant_obj in constraint.task_relevant_objects:
            # 得到一个 4x4 的位姿矩阵（Pose Matrix）
            pose = env.get_obj_pose(task_relevant_obj)
            # 得到一个几何中心到刚体中心变换矩阵
            transform = env.get_obj_geom_transform(task_relevant_obj)
            constraint.update_constraint(pose, transform, task_relevant_obj)

    def motion_plan(
        self,
        obs: Dict[str, Any],
        future_configurations: Optional[np.ndarray] = None,
        future_actions: Optional[np.ndarray] = None,
        motion_planner_type: Literal[
            "eef_interp", "sampling", "curobo", "eef_interp_mink", "eef_interp_curobo"
        ] = "eef_interp",  # from motion_planner itself
        manual_clip_joint_limits: bool = False,
        visualize_failures: bool = False,
    ) -> PlanningResult:
        """Motion plan to the start of the constraint.
        生成一条从当前位置到目标位置的运动轨迹。

        Args:
            obs: 当前的观测数据（包含机器人关节角度、末端位置等）。
            future_configurations / future_actions: 目标构型/目标状态。也就是规划的终点。
            motion_planner_type: 指定使用哪种规划算法（如 curobo, eef_interp）。

        Gripper action:
            - either have an object welded during motion planning -> copy gripper action
            - or no object welded during motion planning -> need to reach a certain target qpos collision free.
                1) harder, more correct: motion plan the fingers too
                2) easier, will lead to failures, and less correct: let the gripper action always to opening action.
        代码逻辑：作者最终选择了简化的启发式策略（Heuristics）：
            正在抓东西（有 Weld） -> 闭合夹爪 (action=1)。
            没抓东西 -> 张开夹爪 (action=-1)。
        If grasping object now, close gripper, else open gripper. TODO(klin): enable end effector motion planning.
        """
        assert future_configurations is not None or future_actions is not None, (
            "At least one of future_configurations"
            "or future_actions is required for motion planning"
        )
        q_start = obs["robot_q"]
        if manual_clip_joint_limits:
            # 如果开启了 manual_clip，会把当前的关节角度强行限制在物理极限范围内。
            # perhaps have a threshold to see if clipping is valid if eef pos doesn't change much
            q_start = np.clip(
                q_start,
                self.robot_configuration.model.jnt_range[:, 0][
                    self.robot_configuration.robot_idxs
                ],
                self.robot_configuration.model.jnt_range[:, 1][
                    self.robot_configuration.robot_idxs
                ],
            )
        X_ee_goal = np.eye(4)
        actions = []
        qs = []

        # future_actions的格式是[x, y, z, rx, ry, rz]（位置+轴角）
        # 构造目标位姿矩阵
        X_ee_goal[:3, 3] = future_actions[0][:3]
        X_ee_goal[:3, :3] = quat2mat(axisangle2quat(future_actions[0][3:6]))
        # 构造起点位姿矩阵
        X_ee_start = np.eye(4)
        X_ee_start[:3, 3] = obs["robot0_eef_pos"]
        X_ee_start[:3, :3] = quat2mat(obs["robot0_eef_quat_site"])
        # 这里这样转换的原因是：
        #  目标规划器的输入是4*4变换矩阵
        #  而actions的值是[x, y, z, rx, ry, rz]（位置+轴角）
        if motion_planner_type == "eef_interp":
            # TODO: handle gripper actions somewhere; depends on if welding object or planning to grasp
            # if planning to close, can start off as either open or closed or keep same
            # post-process the result to get the actions. Generate actions at 20Hz
            # 规划：简单插值，直接在起点和终点之间连一条直线，切成若干份
            interp_pos, interp_quat_xyzw = self.motion_planner.plan(
                X_ee_start, X_ee_goal
            )
            # 格式回转：把切分好的每一步矩阵，变回actions格式
            for pos, quat_xyzw in zip(interp_pos, interp_quat_xyzw):
                action = np.zeros(7)
                action[:3] = pos
                action[3:6] = quat2axisangle(quat_xyzw)
                actions.append(action)

            motion_plan_result = True
        elif motion_planner_type == "eef_interp_mink":
            # 基于逆运动学的简单差值
            # eef_wxyz_xyzs： 规划出的末端执行器（EEF）位姿列表。
            # 格式是 [w, x, y, z, pos_x, pos_y, pos_z]（7维向量）。
            _, eef_wxyz_xyzs = self.motion_planner.plan(
                X_ee_start,
                X_ee_goal,
                q_start[: self.motion_planner.dof],  # 当前机器人的全身关节状态
                self.robot_configuration,  # 传入机器人模型
                # 这是一个硬编码的关节角度数组，代表一个舒适姿态
                # 当机器人去抓一个东西时，手的位置定了，但胳膊肘可以朝上、朝下或朝外（冗余自由度）。
                # Mink 求解器会利用这个参数，在所有可能的解中，选择最靠近这个舒适姿态的解。防止机器人胳膊肘乱飞或者把自己扭成麻花。
                q_retract=np.array(
                    [
                        0,
                        np.pi / 16.0,
                        0.00,
                        -np.pi / 2.0 - np.pi / 3.0,
                        0.00,
                        np.pi - 0.2,
                        np.pi / 4,
                        0,
                        0,
                    ]
                ),
            )
            for wxyz_xyz in eef_wxyz_xyzs:
                action = np.zeros(7)
                action[:3] = wxyz_xyz[4:]
                action[3:6] = quat2axisangle(np.roll(wxyz_xyz[:4], -1))
                actions.append(action)

            motion_plan_result = True
        elif motion_planner_type == "mink_ik":
            # see https://chatgpt.com/c/670f6144-4da8-8009-94a2-da7e014dacc4 # TODO for 10/16
            pass
        elif motion_planner_type == "curobo":
            # curobo可以避障
            batch_size = 1  # 一次只规划一条轨迹。
            retime_to_target_vel = True # 重定时（Retiming）。规划器生成的原始轨迹通常只是几何路径。这个开关告诉规划器：请根据机器人的速度和加速度限制，重新计算时间戳，生成一条物理上可执行的轨迹。
            # need to convert X_ee_goal into the correct frame! We may be lucky if the frame is the same
            self.motion_planner.update_env(
                obs,
                self.env.env,
                env_type="mujoco",
                attach_object_names=obs["attached_object_names"],
                batch_size=batch_size,
            )
            if self.motion_planner.goal_type == "pose_wxyz_xyz":
                # this method gets the curobo goal
                # TODO(klin): need to also get the X_ee that works for the mink planner!
                kin_state = self.motion_planner.motion_gen.ik_solver.fk(
                    torch.tensor(
                        future_configurations[0], device="cuda", dtype=torch.float32
                    )
                )
                ee_goal_curobo = torch.cat(
                    [kin_state.ee_quaternion, kin_state.ee_position], dim=-1
                )[0]
                goal = ee_goal_curobo
            elif self.motion_planner.goal_type == "joint":
                goal = future_configurations[0][: self.motion_planner.dof]
            qs, motion_plan_result = self.motion_planner.plan(
                q_start[: self.motion_planner.dof],
                goal=goal,
                goal_type=self.motion_planner.goal_type,
                batch_size=batch_size,
                retime_to_target_vel=retime_to_target_vel,
                robot_configuration=self.robot_configuration,
            )

            # check that X_base_goal is the same as X_ee_goal
            if qs is None or not motion_plan_result.success.any().item():
                self.robot_configuration.update(q_start)
                if visualize_failures:
                    create_passive_viewer(
                        self.robot_configuration.model, self.robot_configuration.data
                    )
                    self.env.env.env.robots[0].set_robot_joint_positions(
                        future_configurations[0][: self.motion_planner.dof]
                    )
                    self.env.env.env.robots[0].set_gripper_joint_positions(
                        future_configurations[0][-2:]
                    )
                    self.robot_configuration.update(
                        np.concatenate([qs[0], q_start[-2:]])
                    )
                    self.robot_configuration.update(
                        np.concatenate([qs[-1], q_start[-2:]])
                    )
                    self.robot_configuration.update(future_configurations[0])
                    create_passive_viewer(
                        self.robot_configuration.model, self.robot_configuration.data
                    )
                return PlanningResult(
                    actions=None, success=False, robot_configurations=None
                )

            if self.viz_robot_kpts:
                for q in qs:
                    visualize_robot_configuration(
                        self.robot_configuration,
                        self.env,
                        np.concatenate([q, q_start[-2:]]),
                    )
                    time.sleep(0.1)

            for q in qs:
                self.robot_configuration.update(np.concatenate([q, np.zeros(2)]))
                # get the eef pose from the robot configuration
                eef_pose = self.robot_configuration.get_transform_frame_to_world(
                    "gripper0_right_grip_site", "site"
                ).as_matrix()
                # update action to work w/ expected format
                action = np.zeros(7)
                action[:3] = eef_pose[:3, 3]
                # TODO(klin): use the transformed eef pose as the action if no scaling etc actions.
                action[3:6] = quat2axisangle(R.from_matrix(eef_pose[:3, :3]).as_quat())
                actions.append(action)

            if len(obs["attached_object_names"]) == 0:
                # future constraint has no objects attached: currently not grasping --- open gripper
                for action in actions:
                    action[6] = -1
            else:  # objects attached --- close gripper
                for action in actions:
                    action[6] = 1
        elif motion_planner_type == "eef_interp_curobo":
            a = time.time()
            batch_size = 1
            # need to convert X_ee_goal into the correct frame! We may be lucky if the frame is the same
            self.motion_planner.update_env(
                obs,
                self.env.env,
                env_type="mujoco",
                attach_object_names=obs["attached_object_names"],
                batch_size=batch_size,
            )
            b = time.time()
            logging.info(f"update_env time: {b-a}")
            kin_state = self.motion_planner.curobo_planner.motion_gen.ik_solver.fk(
                torch.tensor(
                    future_configurations[0], device="cuda", dtype=torch.float32
                )
            )
            ee_goal_curobo = torch.cat(
                [kin_state.ee_quaternion, kin_state.ee_position], dim=-1
            )[0]
            qs, motion_plan_result = self.motion_planner.plan(
                X_ee_start,
                X_ee_goal,
                self.robot_configuration,
                q_start=q_start[: self.motion_planner.dof],
                q_goal=future_configurations[0][: self.motion_planner.dof],
                retract_q=np.array(
                    [
                        0,
                        np.pi / 16.0,
                        0.00,
                        -np.pi / 2.0 - np.pi / 3.0,
                        0.00,
                        np.pi - 0.2,
                        np.pi / 4,
                        0,
                        0,
                    ]
                ),
                q_gripper_start=q_start[self.motion_planner.dof :],
                ee_goal_curobo=ee_goal_curobo,
                retract_q_weights=np.array([2, 2, 1, 1, 1, 0.1, 0.1, 0, 0]),
                curobo_goal_type=self.motion_planner.curobo_planner.goal_type,
                eef_frame_name="gripper0_right_grip_site",
                eef_frame_type="site",
            )

            # check that X_base_goal is the same as X_ee_goal
            if (
                qs is None
                or (
                    not isinstance(motion_plan_result, bool)
                    and not motion_plan_result.success.any().item()
                )
                or (
                    isinstance(motion_plan_result, bool) and motion_plan_result is False
                )
            ):
                self.robot_configuration.update(q_start)
                if visualize_failures:
                    create_passive_viewer(
                        self.robot_configuration.model, self.robot_configuration.data
                    )
                    self.env.env.env.robots[0].set_robot_joint_positions(
                        future_configurations[0][: self.motion_planner.dof]
                    )
                    self.env.env.env.robots[0].set_gripper_joint_positions(
                        future_configurations[0][-2:]
                    )
                    self.robot_configuration.update(
                        np.concatenate([qs[0], q_start[-2:]])
                    )
                    self.robot_configuration.update(
                        np.concatenate([qs[-1], q_start[-2:]])
                    )
                    self.robot_configuration.update(future_configurations[0])
                    create_passive_viewer(
                        self.robot_configuration.model, self.robot_configuration.data
                    )
                return PlanningResult(
                    actions=None,
                    success=motion_plan_result.success.item()
                    if not isinstance(motion_plan_result, bool)
                    else motion_plan_result,
                    robot_configurations=qs,
                )

            if self.viz_robot_kpts:
                for q in qs:
                    visualize_robot_configuration(
                        self.robot_configuration,
                        self.env,
                        np.concatenate([q, q_start[-2:]]),
                    )
                    time.sleep(0.1)

            for q in qs:
                try:
                    self.robot_configuration.update(np.concatenate([q, np.zeros(2)]))
                except Exception as e:
                    print(e)
                    import ipdb

                    ipdb.set_trace()
                    self.robot_configuration.update(np.concatenate([q, np.zeros(2)]))
                # get the eef pose from the robot configuration
                eef_pose = self.robot_configuration.get_transform_frame_to_world(
                    "gripper0_right_grip_site", "site"
                ).as_matrix()
                # update action to work w/ expected format
                action = np.zeros(7)
                action[:3] = eef_pose[:3, 3]
                # TODO(klin): use the transformed eef pose as the action if no scaling etc actions.
                action[3:6] = quat2axisangle(R.from_matrix(eef_pose[:3, :3]).as_quat())
                actions.append(action)

            if len(obs["attached_object_names"]) == 0:
                # future constraint has no objects attached: currently not grasping --- open gripper
                for action in actions:
                    action[6] = -1
            else:  # objects attached --- close gripper
                for action in actions:
                    action[6] = 1

        return PlanningResult(
            actions=actions,  # 给仿真环境或真机执行的 笛卡尔空间指令（末端位置 + 姿态 + 夹爪）。
            success=motion_plan_result.success.item()
            if motion_plan_result is not True
            else True,
            robot_configurations=qs,  # 规划器内部计算出的 关节角度（Joint Positions）。
        )

    def solve_constraint(
        self,
        constraint: Constraint,
        obs: Dict[str, Any],
        dummy_actions: bool = False,
        copy_constraint_actions: bool = True,
        ik_type: Literal["kpts_to_robot_q", "kpts_to_eef_to_q"] = "kpts_to_eef_to_q",
        fallback_to_kpts_to_robot_q: bool = True,
        solve_first_n_steps: Optional[List[int]] = None,
        # currently, no optimization for kpts_to_eef
    ) -> PlanningResult:
        """Generate a plan based on the updated constraint info.
        给定一个约束和当前环境状态，请计算出连续的机器人动作，满足该约束。
        Args:
            dummy_actions: 调试用，是否生成假动作
            copy_constraint_actions: bool = True, # 策略：是否直接抄演示的动作（作为初始猜测）
            solve_first_n_steps: first n steps for solve constraint for.
                Defaults to None, meaning all steps. Useful if only solving for e.g. first step.
            ik_type: type of IK solver to use.
                - kpts_to_robot_q: [single stage] 直接计算关节。solve IK directly from keypoints to robot q.
                - kpts_to_eef_to_q: [two stage] 先算手，再计算关节。solve IK from keypoints to eef and then from eef to robot q.
                    In practice, currently, doing eef to q directly. TODO(klin): add kpts to eef.
            fallback_to_kpts_to_robot_q: 备选方案：如果两步走失败了，是否尝试一步走。if ik_type="kpts_to_eef_to_q" fails, fallback to "kpts_to_robot_q".
        """
        # Implement the logic for solving the constraint and generating a plan
        if dummy_actions:
            # stay at the current eef pose
            # 生成一堆静止不动的假动作用于测试
            actions = np.zeros((30, 7))
            target_pos = obs["robot0_eef_pos"]
            actions[:, :3] = target_pos  # 全部维持一个位置
            target_quat_xyzw = obs["robot0_eef_quat_site"]
            target_axis_angle = quat2axisangle(target_quat_xyzw)
            actions[:, 3:6] = target_axis_angle  # 全部维持一个角度
            DUMMY_GRIPPER_ACTION = 0.1
            actions[:, 6] = DUMMY_GRIPPER_ACTION  # gripper action，全部维持一个夹爪开合程度
            actions = actions  # + np.random.normal(0, 0.1, actions.shape)

        if copy_constraint_actions:
            # 以原始演示（Demo）中的动作作为求解器的初始猜测（Initial Guess）
            actions = constraint.src_action

        # 为了防止调试用的可视化操作破坏真实的仿真状态
        if self.viz_robot_kpts:
            # record env's current state
            state = self.env.env.get_state()
            # 我们只关心动态数据（位置/速度）。XML 模型是静态的，不需要保存，带着它只会浪费内存和拖慢速度。
            state.pop("model")

        # use the current gripper obj transform to attach the object to the gripper
        # 更新机器人模型和机械手模型
        self._update_robot_configuration(
            self.robot_configuration,
            eef_configuration=self.eef_configuration,
            env=self.env,
            obs=obs,
            constraint=constraint,
        )

        qs = []  # 存放计算出来的关节角度轨迹
        actions = []  # 存放转换后的笛卡尔空间动作。assume action is eef_link_frame
        # let's use the original pose and get the updated pose for now?

        # solve optimization problem to get actions
        # 获取机器人当前时刻的关节角度。
        prev_q = self.robot_configuration.q[self.robot_configuration.robot_idxs]
        # two actuated objects case --- can directly use their original poses to get keypoints
        # one actuated object (either arm or object) case: use the non-actuated object as the anchor
        #   if arm: the optimize hand-pose + gripper config first
        #   if object: optimize object pose, then set the hand+gripper pose accordingly -- the key is knowing what to optimize in robot_configuration
        #       in both cases, need to 1) detect/remove joint between object and robot 2) optimize over the pose
        #       simplification --- can always optimize over robot eef pose (and optionally gripper qpos) for both arm/object being actuated I believe, until doing in hand manipulation
        obs["env"] = self.env
        X_obj0_transf, X_obj1_transf, X_eef_transf = None, None, None
        X_obj0_transf, X_obj1_transf, X_eef_transf = find_best_transform(
            constraint, obs, self.robot_configuration, self.eef_configuration
        )
        solve_first_n_steps: int = (
            solve_first_n_steps
            if solve_first_n_steps is not None
            else len(constraint.timesteps)
        )
        assert (
            solve_first_n_steps <= len(constraint.timesteps)
        ), "constraint_steps should be less than or equal to the number of timesteps in the constraint"
        for t_idx in range(solve_first_n_steps):
            # # how to add in symmetry here? if doing discree option per symmetry, would get exponential explosion
            if constraint.constraint_type == "robot-object":
                link_to_kpts = {
                    link_name: keypoints[t_idx]
                    for link_name, keypoints in constraint.keypoints_robot_link_frame.items()
                }
                # Get scaled keypoints
                task_relev_obj = constraint.obj_names[0]
                # Get the scale factor of the demo object and the current object
                demo_obj_geoms_size: Dict[str, np.ndarray] = (
                    constraint.src_obj_geoms_size[task_relev_obj][t_idx]
                )
                current_obj_geoms_size: Dict[str, np.ndarray] = obs[
                    f"{task_relev_obj}_geoms_size"
                ]
                # Use the helper function to get the scale factor
                scale_factor = (
                    get_scale_factor(demo_obj_geoms_size, current_obj_geoms_size)
                    if not self.ignore_scaling
                    else 1.0
                )
                # Scale the original keypoints to the current object's scale
                P_obj_target = (
                    constraint.keypoints_obj_frame[task_relev_obj][t_idx] * scale_factor
                )
                P_W_target = transform_keypoints(
                    P_obj_target, obs[f"{constraint.obj_names[0]}_pose"] @ X_obj0_transf
                )[:, :3]
            elif constraint.constraint_type == "object-object":
                # only keep keypoints attached to the kinematically 'actuated' object
                link_to_kpts = {
                    link_name: keypoints[t_idx]
                    for link_name, keypoints in constraint.keypoints_obj_frame.items()
                    if link_name in constraint.obj_to_parent_attachment_frame
                    and constraint.obj_to_parent_attachment_frame[link_name] is not None
                }
                # get object keypoints in world frame: will be used in optimization q* = argmin_q ||P_W_goal - P_W_robot_kp(q)||^2
                P_obj_target = np.array(
                    constraint.keypoints_obj_frame[constraint.obj_names[1]][t_idx]
                )
                P_W_target = transform_keypoints(
                    P_obj_target, obs[f"{constraint.obj_names[1]}_pose"] @ X_obj1_transf
                )[:, :3]  # s

                use_weld_heuristic_action: bool = True
                if use_weld_heuristic_action:
                    constraint.src_gripper_action[t_idx] = (
                        1  # hack for now to test if we needed to match actions rather than states
                    )

            flip_eef = not np.allclose(X_eef_transf, np.eye(4), rtol=1e-5, atol=1e-8)
            self.robot_configuration.set_keypoints(link_to_kpts, flip_eef=flip_eef)
            self.eef_configuration.set_keypoints(link_to_kpts, flip_eef=flip_eef)

            if self.viz_robot_kpts:
                # set mocap keypoints to the target keypoints
                set_mocap_pos_and_update_viewer(
                    obs["env"],
                    self.robot_configuration.get_keypoints(),
                    update_viewer=False,
                )
                render_image(
                    obs["env"].env.env.sim.model._model,
                    obs["env"].env.env.sim.data._data,
                    image_prefix="kpts-rob",
                )
                set_mocap_pos_and_update_viewer(
                    obs["env"], P_W_target, update_viewer=False
                )
                set_mocap_pos_and_update_viewer(
                    obs["env"], P_W_target, update_viewer=False
                )
                render_image(
                    obs["env"].env.env.sim.model._model,
                    obs["env"].env.env.sim.data._data,
                    image_prefix="kpts-obj-1",
                )
                set_mocap_pos_and_update_viewer(
                    obs["env"], P_W_target[2:], update_viewer=False
                )
                render_image(
                    obs["env"].env.env.sim.model._model,
                    obs["env"].env.env.sim.data._data,
                    image_prefix="kpts-obj-2",
                )
            # potentially merge eef des computation w/ joint q computation
            X_W_eef_des = compute_X_W_eef_des(
                constraint,
                obs,
                self.robot_configuration,
                self.eef_configuration,
                X_obj0_transf=X_obj0_transf,
                X_obj1_transf=X_obj1_transf,
                X_eef_transf=X_eef_transf,
                src_t_idx=t_idx,
                ignore_scaling=self.ignore_scaling,
            )
            q_eef_config = np.zeros(7 + 2)
            q_eef_config[:3] = X_W_eef_des[:3, 3]
            q_eef_config[3:7] = mat2quat(X_W_eef_des[:3, :3])
            q_eef_config[7:] = constraint.src_obs["robot0_gripper_qpos"][t_idx]
            self.eef_configuration.update(q_eef_config)
            # get the action
            action = np.zeros(7)
            action[:3] = X_W_eef_des[:3, 3]
            action[3:6] = quat2axisangle(np.roll(mat2quat(X_W_eef_des[:3, :3]), -1))
            action[6] = constraint.src_gripper_action[t_idx].item()
            optimized_q = optimize_robot_configuration(
                self.robot_configuration,
                self.eef_configuration,
                retract_q=np.array(
                    [
                        0,
                        np.pi / 16.0,
                        0.00,
                        -np.pi / 2.0 - np.pi / 3.0,
                        0.00,
                        np.pi - 0.2,
                        np.pi / 4,
                        0,
                        0,
                    ]
                ),
                update_gripper_qpos=False,
                retract_q_weights=np.array([2, 2, 1, 1, 1, 0.1, 0.1, 0, 0]),
                # prefer the 'elbow' up pose: domain knowledge to avoid singularities
                start_q=prev_q,
                eef_interp_mink_mp=self.eef_interp_mink_motion_planer,
                optimization_type="scipy" if t_idx == 0 else "scipy",
                verbose=self.verbose,
            )

            if optimized_q is None:
                if self.verbose:
                    logging.info(
                        "Optimization failed. Usually occurs when a previous stage wasn't executed correctly or when. "
                        "the optimization problem is ill-posed due to e.g. kinematics."
                    )
                if fallback_to_kpts_to_robot_q:
                    logging.info("Falling back to ik_type=kpts_to_robot_q.")
                else:
                    return PlanningResult(
                        actions=None,
                        robot_configurations=None,
                        success=False,
                    )
            else:
                optimized_q[7:] = constraint.src_obs["robot0_gripper_qpos"][
                    t_idx
                ]  # hack for now to test if we needed to match actions rather than states
            if (
                optimized_q is None and fallback_to_kpts_to_robot_q
            ) or ik_type == "kpts_to_robot_q":
                logging.info(f"Falling back to IK on keypoints for t_idx: {t_idx}")
                # TODO: set up IK optimization for eef pose. For now, directly provide eef pose for curobo's expected frame
                start = time.time()
                if t_idx == 0:
                    # need to get the keypoint of the SquareNut to the peg
                    prev_q = optimize_robot_configuration_kp(
                        self.robot_configuration,
                        P_W_target,
                        prev_q,
                        max_iter=self.max_iter + 100,
                        verbose=True,
                    )
                    prev_q = optimize_robot_configuration_kp(
                        self.robot_configuration,
                        P_W_target,
                        prev_q,
                        max_iter=self.max_iter + 100,
                        verbose=True,
                    )
                optimized_q = optimize_robot_configuration_kp(
                    self.robot_configuration,
                    P_W_target,
                    prev_q,
                    max_iter=self.max_iter,
                    verbose=True,
                    global_opt=False,
                )
                logging.info(
                    f"Optimization t_idx: {t_idx} took {time.time() - start} seconds"
                )

            if self.viz_robot_kpts:
                set_mocap_pos_and_update_viewer(self.env, P_W_target)
                visualize_robot_configuration_and_keypoints(
                    self.robot_configuration,
                    self.env,
                    optimized_q,
                    constraint=constraint,
                )

            qs.append(optimized_q)

            self.robot_configuration.update(optimized_q)
            # get the eef pose from the robot configuration
            eef_pose = self.robot_configuration.get_transform_frame_to_world(
                "gripper0_right_grip_site", "site"
            ).as_matrix()
            # update action to work w/ expected format
            action = np.zeros(7)
            action[:3] = eef_pose[:3, 3]
            # TODO(klin): use the transformed eef pose as the action if no scaling etc actions.
            action[3:6] = quat2axisangle(R.from_matrix(eef_pose[:3, :3]).as_quat())
            action[6] = constraint.src_gripper_action[
                t_idx
            ].item()  # TODO: do better than copying the gripper action
            actions.append(action)

        if self.viz_robot_kpts:
            # reset the env to the original state
            self.env.env.reset_to(
                state
            )  # ideally these methods are inside self.env itself
            self.env.env.env.sim.forward()
            self.env.env.env.viewer.update()

        for behavior in constraint.post_constraint_behavior:  # heuristic
            if behavior == "lift":
                actions.extend(generate_lift_actions(actions[-1]))
            elif behavior == "open_gripper":
                actions.extend(generate_open_gripper_actions(actions[-1]))
            elif behavior == "close_gripper":
                actions.extend(generate_close_gripper_actions(actions[-1]))
            # TODO also update qs based on the extra actions

        actions = [np.array(a) for a in actions]
        return PlanningResult(
            actions=actions,
            robot_configurations=qs,
            success=True,
        )

    def advance_constraint(self):
        """Advance to the next constraint."""
        self.current_constraint_idx += 1
        self.check_constraint = False

    def is_constraint_satisfied(self, obs: Dict) -> bool:
        """Check if the current constraint is satisfied in the environment.

        Some reasons for non-satisfaction:

        1) epsilon threshold too small
        2) couldn't track the plan well enough due to e.g. collision
        """
        return True  # dummy return True for now
        # constraint = self.current_constraint()
        # self.update_constraint(env, constraint)
        # return constraint.is_satisfied()

    def _update_robot_configuration(
        self,
        robot_configuration: IndexedConfiguration,
        eef_configuration: Optional[IndexedConfiguration] = None,
        env: Optional[CPEnv] = None,
        obs: Dict[str, Any] = None,
        constraint: Optional[Constraint] = None,
        do_remove_free_joints: bool = False,
        q_eef: Optional[np.ndarray] = None,
    ) -> None:
        """
        根据当前的观测和约束更新机器人模型和机械手模型
        Update the robot configuration based on the current observation and constraint.

        TODO: test implementation on bimanual welding
        """
        # 获取当前真实的机器人关节角度
        q = obs["robot_q"]
        # 更新机器人模型的关节状态
        robot_configuration.update(q)
        xml_string = self.original_xml
        update_model: bool = False

        # update the robot configuration: include robot and welded objects
        for obj_name in constraint.obj_names:
            # get weld transform between parent (robot) frame and object i.e. X_robot_obj = X_W_robot^-1 @ X_W_obj
            # 检查约束：如果这个物体没有定义“父级挂载点”（即没被抓住），就跳过。
            if constraint.obj_to_parent_attachment_frame is None:
                continue
            # 获取父级坐标系名字（例如 "gripper0_right_eef"）
            obj_parent_frame = constraint.obj_to_parent_attachment_frame.get(
                obj_name, None
            )
            if obj_parent_frame is None:
                continue
            # 物体在世界坐标系下的位姿
            X_W_obj = obs[f"{obj_name}_pose"]
            # 机器人的挂载点（手）在世界坐标系下的位姿（从正运动学算）
            X_W_robot = robot_configuration.get_transform_frame_to_world(
                obj_parent_frame, "body"
            ).as_matrix()
            # 物体相对于手的位姿
            X_robot_obj = np.linalg.inv(X_W_robot) @ X_W_obj
            # modify update the robot configuration model
            # 调用工具函数 weld_frames。它会在 XML 文本层面，把该物体的 body 节点
            # 移动到机器人的 body 节点下面，并固定住相对位置（X_robot_obj）。
            # 这样在物理引擎看来，它们就是一个刚体了。
            xml_string = weld_frames(
                xml_string, obj_parent_frame, obj_name, X_robot_obj
            )
        # always update model for now
        update_model = True  # TODO(klin): refactor to avoid always updating the model

        model = mujoco.MjModel.from_xml_string(xml_string) if update_model else None
        # 更新机器人模型的静态结构，以及关节角度
        robot_configuration.update(q, model=model)

        # 把 真实仿真环境 (self.env) 里那一刻的所有物理细节，原封不动地 拷贝 到 内部模型 (robot_configuration) 里。
        # 动态数据（如物体速度、加速度、摩擦力系数、当前的接触力等）
        sync_mjmodel_mjdata(
            self.env.env.env.sim.model._model,
            self.env.env.env.sim.data._data,
            robot_configuration.model,
            robot_configuration.data,
            verbose=False,
        )

        # update eef-configuration using the robot configuration
        # 计算全身模型中，末端执行器（Gripper Body）在世界坐标系下的位姿。
        # 返回的 wxyz_xyz 是一个 7 维向量：[w, x, y, z, pos_x, pos_y, pos_z]
        wxyz_xyz = robot_configuration.get_transform_frame_to_world(
            "gripper0_right_right_gripper", "body"
        ).wxyz_xyz
        # 更新状态向量“手心在哪里（Pos）”、“手朝向哪里（Rot）”以及“手指张开多少（Gripper Q）”
        eef_configuration.update(np.concatenate([wxyz_xyz[4:], wxyz_xyz[:4], q[-2:]]))
        # 不带参数调用 update()，通常是强制触发一次正运动学（Forward Kinematics）计算，确保所有的缓存数据（如连杆的全局位置、雅可比矩阵等）都是最新的、同步完成的状态。
        robot_configuration.update()
        if eef_configuration is not None:
            # 重建并同步那个“只包含手和物体”的局部模型（EEF Configuration）
            # 提取夹爪角度 get original gripper qpos
            qpos_gripper = robot_configuration.q[robot_configuration.robot_idxs][
                -2:
            ]  # hack
            # 计算夹爪相对于世界坐标系的位姿
            q_gripper = robot_configuration.get_transform_frame_to_world(
                "gripper0_right_right_gripper", "body"
            ).wxyz_xyz
            # convert to xyz_wxyz
            # 重组状态向量
            # q_gripper[4:]：取位置 [px, py, pz]。
            # q_gripper[:4]：取四元数 [w, x, y, z]。
            # qpos_gripper：取手指关节角度。
            q_gripper = np.concatenate([q_gripper[4:], q_gripper[:4], qpos_gripper])
            # 这个 xml_string 是刚刚经过 weld_frames 修改过的（包含了被焊在手上的物体）。
            # 得到一个新的 IndexedConfiguration 对象。这个对象里的“手”上，现在长着一个“物体”。
            new_eef_configuration = get_eef_configuration(
                xml_string,
                robot_configuration.model,
                robot_configuration.data,
                gripper_joint_names=env.env.env.robots[0]
                .robot_model.grippers["robot0_right_hand"]
                .joints,
            )
            self.eef_configuration = new_eef_configuration
            self.eef_configuration.update(q_gripper)

            assert np.allclose(
                robot_configuration.get_transform_frame_to_world(
                    "gripper0_right_right_gripper", "body"
                ).wxyz_xyz,
                self.eef_configuration.get_transform_frame_to_world(
                    "gripper0_right_right_gripper", "body"
                ).wxyz_xyz,
            ), "eef configuration should match robot configuration"

    def get_action(self, obs: Dict[str, Any]) -> Dict[str, Union[np.ndarray, str]]:
        """维护了一个动作队列 (action_queue)。外部环境每调用一次 get_action，它就吐出一个动作。如果队列空了，它就会触发耗时的计算（解算约束 + 运动规划）来生成下一批动作。"""
        if self.check_constraint:
            if not self.is_constraint_satisfied(obs):
                logging.info(
                    f"Constraint {self.current_constraint_idx} not satisfied. Continuing for now ..."
                )
            self.advance_constraint()

        if self.done:
            logging.info("Policy is done. No more actions to take.")
            return {"action": None, "failure_type": None}

        # If there are remaining actions in the queue, return the next action
        if not self.action_queue:
            # update robot configuration based on current model
            # Get the current constraint based on the observation or internal tracker
            constraint = self.current_constraint
            # Solve the constraint, and if it fails, return no action
            a = time.time()
            result = self.solve_constraint(
                constraint, obs
            )  # assume abs actions I suppose
            if constraint.duplicate_actions:
                result.robot_configurations = [
                    conf for conf in result.robot_configurations for _ in [0, 1]
                ]
                result.actions = [conf for conf in result.actions for _ in [0, 1]]
            if self.verbose:
                logging.info(f"Solving constraint took {time.time() - a} seconds")
            if not result.success:
                logging.warning(
                    "Failed to solve constraint during solve_constraint. Returning no action."
                )
                return {"action": None, "failure_type": "failed_to_solve_constraint"}

            # If the constraint was solved, motion planning to the start of the constraint; pass info to mp
            obs["attached_object_names"] = (
                constraint.attached_object_names
            )  # pass attachment info to planner via obs
            a = time.time()
            mp_result = self.motion_plan(
                obs,
                result.robot_configurations,
                result.actions,
                motion_planner_type=self.motion_planner.name
                if self.motion_planner is not None
                else None,
                manual_clip_joint_limits=True,
                visualize_failures=False,
            )

            if self.verbose:
                logging.info(f"Motion planning took {time.time() - a} seconds")
            if not mp_result.success:
                # visualize eef config
                # convert actions to eef body frame
                if self.verbose:
                    constraint_pos = result.actions[0][:3]
                    constraint_axis_angle = result.actions[0][3:6]

                    X_W_eef_curr = self.eef_configuration.get_transform_frame_to_world(
                        "gripper0_right_right_gripper", "body"
                    ).as_matrix()
                    X_W_eefsite_curr = (
                        self.eef_configuration.get_transform_frame_to_world(
                            "gripper0_right_grip_site", "site"
                        ).as_matrix()
                    )
                    X_eefsite_eef = np.linalg.inv(X_W_eefsite_curr) @ X_W_eef_curr
                    constraint_rot_mat = quat2mat(axisangle2quat(constraint_axis_angle))
                    X_W_eefsite = make_pose(constraint_pos, constraint_rot_mat)
                    X_W_eef = X_W_eefsite @ X_eefsite_eef
                    # update end_effector_configuration for collision checking on the current obstacle poses
                    xyz_wxyz = np.concatenate(
                        [
                            X_W_eef[:3, 3],
                            mat2quat(X_W_eef[:3, :3]),
                            result.robot_configurations[0][7:],
                        ]
                    )
                    self.eef_configuration.update(xyz_wxyz)
                    create_passive_viewer(
                        self.eef_configuration.model,
                        self.eef_configuration.data,
                        rate=10,
                    )
                    # view robot configuration
                    # update robot config according to results
                    self.robot_configuration.update(result.robot_configurations[0])
                    create_passive_viewer(
                        self.robot_configuration.model,
                        self.robot_configuration.data,
                        rate=10,
                    )
                logging.error(
                    "Failed to solve constraint during motion_plan. Returning no action."
                )
                return {"action": None, "failure_type": "motion_planning_failure"}

            # Store the planned actions for later execution
            self.action_queue = mp_result.actions + result.actions
            self.curr_motion_plan_steps = len(mp_result.actions)
            self.curr_constraint_steps = len(result.actions)
            self.new_constraint = True
        else:
            self.new_constraint = False

        # Return the first action from the queue
        action = self.action_queue.pop(0)
        if not self.action_queue:  # exhausted the queue so check the constraint
            self.check_constraint = True

        return {
            "action": action,
            "failure_type": None,
        }


def anonymize_model_file_paths(xml_str: str) -> str:
    if xml_str.lstrip().startswith("<?xml"):
        parser = ET.XMLParser(remove_blank_text=True)
        root = ET.fromstring(xml_str.encode("utf-8"), parser=parser)
    else:
        root = ET.fromstring(xml_str)

    for elem in root.iter():
        if "file" in elem.attrib:
            original = elem.attrib["file"]
            for key in ["cpgen", "mimicgen", "robosuite"]:
                # ordering matters: mimicgen paths might have robosuite, but not vice-versa
                if key in original:
                    idx = original.index(key)
                    elem.attrib["file"] = "/path/to" + original[idx - 1 :]
                    break

    return ET.tostring(root, pretty_print=True, encoding="unicode")

@dataclass
class SystemNoiseConfig:
    # motion_segment_noise_magnitude：在 “自由移动阶段（Motion Segment）” 时的噪声大小
    motion_segment_noise_magnitude: Union[float, List[float]] = field(
        default_factory=lambda: [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    )
    """Magnitude of noise to add to robot actions during motion planning segments"""
    # constraint_segment_noise_magnitude：在 “受限交互阶段（Constraint Segment）” 时的噪声大小
    constraint_segment_noise_magnitude: Union[float, List[float]] = field(
        default_factory=lambda: [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    )
    """Magnitude of noise to add to robot actions during constraint execution segments"""
    # skip_noise_for_last_stage：是否在最后关头停止抖动
    # 如果最后一步还在抖，很有可能导致物体倒了或者任务判定失败，那这条数据就白跑了。所以这是一个**“稳一手”**的开关。
    skip_noise_for_last_stage: bool = True
    """Whether to skip adding noise during the last constraint segment execution"""

class DemoGenerator:
    def __init__(
        self,
        env: Union[CPEnv, Any],
        policy: CPPolicy,
        use_reset_near_constraint: bool = False,
        constraint_selection_method: Literal[
            "random", "first", "second", "last"
        ] = "random",
        use_reset_to_state: bool = False,
        reset_state_demo_path: Optional[str] = None,
        reset_state_demo_idx: Optional[int] = None,
        demo_src_env: Optional[CPEnv] = None,
        anonymize_model_file_paths: bool = True,
    ):
        self.env = env
        self.policy = policy
        self.use_reset_near_constraint = use_reset_near_constraint
        self.constraint_selection_method = constraint_selection_method
        self.use_reset_to_state = use_reset_to_state
        self.reset_state_demo_path = reset_state_demo_path
        self.reset_state_demo_idx = reset_state_demo_idx
        self.demo_src_env = demo_src_env
        self.anonymize_model_file_paths = anonymize_model_file_paths

    def reset_near_constraint(
        self,
        constraint: Constraint,
        pos_bounds: Tuple[
            Tuple[float, float], Tuple[float, float], Tuple[float, float]
        ],
        max_rot_angle: float,
        randomize_gripper: bool = False,
        verbose: bool = False,
    ) -> Dict[str, Any]:
        """
        Resets the environment near a given source demo constraint state by sampling pose variations.
        Code structure: perhaps this function should be independent of the DemoGenerator class and be
            a separate submodule?
            Collision checking: technically the env should be the one doing the collision checking, however,
            we're separating the robot_configuration from the env at least in policy.

        Args:
            constraint_state (Dict[str, Any]): Desired constraint state containing positions and orientations.
            variation_range (Tuple[float, float]): Min and max bounds for position and rotation variation.
        Returns:
            Dict[str, Any]: New configuration to pass to `self.env.reset_to()`.
        """
        a = time.time()
        # issues: i) won't have examples where the gripper is flipped.
        # ii) can't randomly reset all objects; else stage needs to shift again.
        # Would need to know what reset and what not to reset.
        # overcome issues by resetting to previous successfull state
        src_state_idx = 0
        if constraint.reset_near_random_constraint_state:
            src_state_idx = np.random.randint(
                len(constraint.src_state) * 1 // 5, len(constraint.src_state) * 4 // 5
            )
        self.demo_src_env.env.reset_to(
            {
                "states": constraint.src_state[src_state_idx],
                "model": constraint.src_model_file,
            }
        )

        sync_mjmodel_mjdata(
            self.demo_src_env.env.env.sim.model._model,
            self.demo_src_env.env.env.sim.data._data,
            self.env.env.env.sim.model._model,
            self.env.env.env.sim.data._data,
            verbose=False,
        )
        self.env.env.env.sim.forward()
        obs = self.env.get_observation()

        if constraint.constraint_type == "robot-object":
            # default to randomizing gripper qpos
            gripper_joint_positions = []
            for joint_name in self.env.env.env.robots[0].gripper["right"].joints:
                joint_id = self.env.env.env.sim.model.joint_name2id(joint_name)
                gripper_jnt_range = self.env.env.env.sim.model._model.jnt_range[
                    joint_id
                ]
                gripper_joint_positions.append(
                    np.random.uniform(gripper_jnt_range[0], gripper_jnt_range[1])
                )

            # Set all gripper joint positions at once
            self.env.env.env.robots[0].set_gripper_joint_positions(
                gripper_joint_positions
            )
            obs["robot0_gripper_qpos"] = gripper_joint_positions
            obs["robot_q"] = np.concatenate(
                [obs["robot0_joint_pos"], obs["robot0_gripper_qpos"]]
            )

        self.policy.verbose = False
        result = self.policy.solve_constraint(
            constraint, obs, solve_first_n_steps=1
        )  # assume abs actions I suppose
        if not result.success:
            logging.error(
                "Failed to solve constraint during solve_constraint. Failing to reset near constraint. "
                "TODO: handle this case."
            )
            return {}

        self.policy.verbose = False
        if self.policy.verbose:
            print("visualizing env configuration")
            create_passive_viewer(
                self.env.env.env.sim.model._model,
                self.env.env.env.sim.data._data,
                show_left_ui=False,
                show_right_ui=False,
            )
        sync_mjmodel_mjdata(
            self.env.env.env.sim.model._model,
            self.env.env.env.sim.data._data,
            self.policy.robot_configuration.model,
            self.policy.robot_configuration.data,
            verbose=False,
        )
        self.policy.robot_configuration.update()
        if self.policy.verbose:
            print("visualizing robot configuration")
            create_passive_viewer(
                self.policy.robot_configuration.model,
                self.policy.robot_configuration.data,
                show_left_ui=False,
                show_right_ui=False,
            )
        self.policy._update_robot_configuration(
            self.policy.robot_configuration,
            eef_configuration=self.policy.eef_configuration,
            obs=obs,
            constraint=constraint,
            env=self.env,
        )
        eef_configuration = self.policy.eef_configuration
        self.policy.robot_configuration.update(result.robot_configurations[0])
        X_W_eef_curr = self.policy.robot_configuration.get_transform_frame_to_world(
            "gripper0_right_right_gripper", "body"
        ).as_matrix()
        X_W_eefsite_curr = self.policy.robot_configuration.get_transform_frame_to_world(
            "gripper0_right_grip_site", "site"
        ).as_matrix()
        X_eefsite_eef = np.linalg.inv(X_W_eefsite_curr) @ X_W_eef_curr

        if constraint.constraint_type == "object-object":
            X_W_obj = obs[f"{constraint.obj_names[0]}_pose"]
            X_eef_obj = (
                np.linalg.inv(X_W_eef_curr) @ X_W_obj
            )  # relevant pose between hand and object

        max_samples: int = 100
        for itr in range(max_samples):
            # eef-pose from result.actions[0]; corresponding configuration from result.robot_configurations[0]
            delta_pose_variation = random_pose(pos_bounds, max_rot_angle)
            constraint_pos = result.actions[0][:3]
            constraint_axis_angle = result.actions[0][3:6]
            constraint_rot_mat = quat2mat(axisangle2quat(constraint_axis_angle))
            constraint_pose_site = make_pose(constraint_pos, constraint_rot_mat)
            X_W_eefsite = constraint_pose_site @ delta_pose_variation
            X_W_eef = X_W_eefsite @ X_eefsite_eef
            # update end_effector_configuration for collision checking on the current obstacle poses
            xyz_wxyz = np.concatenate(
                [
                    X_W_eef[:3, 3],
                    mat2quat(X_W_eef[:3, :3]),
                    result.robot_configurations[0][7:],
                ]
            )
            # TODO need to also reset/update the object's pose with delta_pose_variation
            eef_configuration.update(xyz_wxyz)
            model = eef_configuration.model
            data = eef_configuration.data
            robot_geoms = get_subtree_geom_ids_by_group(
                model, model.body("gripper0_right_right_gripper").id, group=0
            )
            # robot_geoms = get_subtree_geom_ids_by_group(model, model.body("robot0_link0").id, group=0)
            body_ids = get_top_level_bodies(
                model, exclude_prefixes=["robot", "gripper"]
            )
            body_names = [get_body_name(model, body_id) for body_id in body_ids]
            non_robot_geoms = [
                geom_id
                for body_id in body_ids
                for geom_id in get_subtree_geom_ids_by_group(model, body_id, group=0)
            ]
            if verbose:
                logging.info(f"top-level body_names: {body_names}")
                logging.info(f"Filtered top-level bodies: {body_ids}")
            # get names related to the robot
            geom_pairs_to_check: List[Tuple] = [(robot_geoms, non_robot_geoms)]
            is_collision = (
                len(check_geom_collisions(model, data, geom_pairs_to_check)) > 0
            )
            if not is_collision:
                if verbose:
                    logging.info(f"Found a valid pose after {itr} samples.")
                # visualize_robot_configuration(self.policy.robot_configuration, self.env, q)
                # visualize the q for sanity check TODO(klin)
                model = self.policy.eef_configuration.model
                data = self.policy.eef_configuration.data
                if self.policy.verbose:
                    create_passive_viewer(
                        model, data, show_left_ui=False, show_right_ui=False
                    )
            elif is_collision:
                if self.policy.verbose:
                    logging.info(f"Collision detected for pose {itr}. Continuing ...")
                    create_passive_viewer(
                        model, data, show_left_ui=False, show_right_ui=False
                    )
                    model = self.policy.eef_configuration.model
                    data = self.policy.eef_configuration.data
                continue

            a = time.time()
            q = optimize_robot_configuration(
                self.policy.robot_configuration,
                eef_target=X_W_eefsite,
                eef_target_type="pose",
                eef_frame_name="gripper0_right_grip_site",
                eef_frame_type="site",
                retract_q=np.array(
                    [
                        0,
                        np.pi / 16.0,
                        0.00,
                        -np.pi / 2.0 - np.pi / 3.0,
                        0.00,
                        np.pi - 0.2,
                        np.pi / 4,
                        0,
                        0,
                    ]
                ),
                retract_q_weights=np.array([2, 2, 1, 1, 1, 0.1, 0.1, 0, 0]),
                update_gripper_qpos=False,
            )
            if verbose:
                logging.info(f"IK took {time.time() - a} seconds")
            if q is None:
                logging.info(f"IK took {time.time() - a} seconds")
                logging.info(
                    "Failed to solve IK for new pose. Failing to reset near constraint. "
                    "EEF pose likely kinematically infeasible. Continuing ..."
                )
                # TODO(klin): should first check if the thing is incollision and then check if IK
                # is possible
                continue

            q[self.policy.motion_planner.dof :] = obs["robot_q"][
                self.policy.motion_planner.dof :
            ]
            obs["robot_q"] = q
            break

        self.policy.robot_configuration.update(q)
        if self.policy.verbose:
            model = self.policy.robot_configuration.model
            data = self.policy.robot_configuration.data
            create_passive_viewer(model, data, show_left_ui=False, show_right_ui=False)
        # TODO: get eef configuration to include the square in correct position
        model = self.policy.eef_configuration.model
        data = self.policy.eef_configuration.data
        model = self.env.env.env.sim.model._model
        data = self.env.env.env.sim.data._data
        self.env.env.env.robots[0].set_robot_joint_positions(
            q[: self.policy.motion_planner.dof]
        )
        self.env.env.env.robots[0].set_gripper_joint_positions(
            q[self.policy.motion_planner.dof :]
        )
        if constraint.constraint_type == "object-object":
            X_W_obj = obs[f"{constraint.obj_names[0]}_pose"]
            X_W_obj_des = X_W_eef @ X_eef_obj
            set_body_pose(
                self.env.env.env.sim.data,
                self.env.env.env.sim.model._model,
                constraint.obj_names[0],
                X_W_obj_des[:3, 3],
                mat2quat(X_W_obj_des[:3, :3]),
            )
            self.env.env.env.sim.forward()

        # TODO: klin: after finding q, also need to set object's pose in the data field.
        # should check if .forward() forwards the simulator too.
        if verbose:
            logging.info(f"Resetting to new pose took {time.time() - a} seconds")
        return self.env.get_observation()  # TODO(klin): check if get_state is correct

    def generate_demo(
        self,
        gen_single_stage_only: bool = False,
        store_single_stage_only: bool = False,
        store_single_stage_max_extra_steps: int = 0,
        system_noise_cfg: SystemNoiseConfig = field(default_factory=SystemNoiseConfig)
    ) -> Dict[str, Any]:
        obss, actions, dones, rewards, infos, states = [], [], [], [], [], []
        constraint_sequence: List[Constraint] = []
        timestep = 0
        obs = self.env.reset()
        state = self.env.get_state()
        if self.use_reset_near_constraint:
            # 选择demo
            self.policy.current_constraint_sequence_idx = random.randint(
                0, len(self.policy.constraint_sequences) - 1
            )  # randomly choose constraint sequence from which to choose a constraint
            # maybe the source constraint method selection can be here
            # 选择demo的阶段
            if self.constraint_selection_method == "random":
                self.policy.current_constraint_idx = random.randint(
                    0, len(self.policy.current_constraint_sequence) - 1
                )
            elif self.constraint_selection_method == "first":
                self.policy.current_constraint_idx = 0
            elif self.constraint_selection_method == "second":
                self.policy.current_constraint_idx = 1
            elif self.constraint_selection_method == "third":
                self.policy.current_constraint_idx = 2
            elif self.constraint_selection_method == "last":
                self.policy.current_constraint_idx = (
                    len(self.policy.current_constraint_sequence) - 1
                )

            # reset_near_constraint 会计算目标位姿，并利用逆运动学（IK）把机器人瞬移过去。
            # pos_bounds定义了X, Y, Z 三个轴的随机扰动范围
            # max_rot_angle设置旋转噪声，允许机器人有最大 45 度 (π/4) 的随机旋转偏差。
            # randomize_gripper=True：随机化夹爪状态。
            obs = self.reset_near_constraint(
                self.policy.current_constraint,
                pos_bounds=(
                    (
                        -self.policy.current_constraint.reset_near_constraint_pos_bound,
                        self.policy.current_constraint.reset_near_constraint_pos_bound,
                    ),
                    (
                        -self.policy.current_constraint.reset_near_constraint_pos_bound,
                        self.policy.current_constraint.reset_near_constraint_pos_bound,
                    ),
                    (
                        -self.policy.current_constraint.reset_near_constraint_pos_bound,
                        self.policy.current_constraint.reset_near_constraint_pos_bound,
                    ),
                ),
                max_rot_angle=np.pi / 4,
                verbose=self.policy.verbose,
                randomize_gripper=True,
            )
            state["states"] = self.env.get_state()["states"]
        elif self.use_reset_to_state:
            # 强制读取指定状态
            demos = load_demos(self.reset_state_demo_path, self.reset_state_demo_idx)
            state["states"] = demos[0].states[0]
            self.env.env.reset_to(state)
            obs = self.env.get_observation()

        model = state[
            "model"
        ]  # temp: use original model file to avoid get_xml() issues
        states.append(state["states"])
        obss.append(obs)
        success = False
        timestep = 0
        failure_type = None
        while True:
            action_dct = self.policy.get_action(obs)
            action = action_dct["action"]
            if action is None:
                # 如果策略算不出动作（比如规划失败）
                failure_type = action_dct["failure_type"]
                if failure_type is not None:  # failured
                    logging.info(f"policy.get_action failed with type: {failure_type}")
                    # # 标记本步奖励为 0（失败）
                    if len(rewards) > 0:
                        rewards[-1] = 0
                    else:
                        rewards.append(0)
                    break
            # 策略说：“我的任务清单全做完了”。
            # 跳出循环，本次生成成功结束。
            if self.policy.done:
                break

            # 这一段逻辑只有在 “切换任务阶段”（比如从移动切换到抓取）的那一瞬间才会触发。
            # 它的目的是为了给生成的数据打上标签，告诉未来的使用者：“这一段数据是在执行什么任务”。
            if self.policy.new_constraint:
                if isinstance(self.env, CPEnvRobomimic):
                    # 将当前物理引擎内存中（MjModel）那些没有关节（Fixed Joint）的物体的最新位置，反向写入到 XML 字符串中。
                    # 因为抓取动作可能改变了物体的父子关系（焊接）
                    model = update_fixed_joint_objects_in_xml(
                        self.env.env.env.sim.model, model
                    )

                # 这里把策略里那个用于规划的 Constraint 对象，复制了一份，
                # 准备作为“数据标签”存入 HDF5。
                curr_constraint = Constraint(
                    obj_names=self.policy.current_constraint.obj_names,
                    obj_to_parent_attachment_frame=self.policy.current_constraint.obj_to_parent_attachment_frame,
                    post_constraint_behavior=self.policy.current_constraint.post_constraint_behavior,
                    constraint_type=self.policy.current_constraint.constraint_type,
                    keypoints_obj_frame_annotation=self.policy.current_constraint.keypoints_obj_frame_annotation,
                    keypoints_robot_link_frame_annotation=self.policy.current_constraint.keypoints_robot_link_frame_annotation,
                    src_model_file=model,  # 记录此时的物理模型
                    reset_near_random_constraint_state=self.policy.current_constraint.reset_near_random_constraint_state,
                )
                # Start = 当前时间步 + 刚才规划花了多少步赶路 (motion_plan_steps)。
                # End   = Start + 预计执行这个约束要花多少步 (constraint_steps)。
                curr_constraint.timesteps = list(
                    range(
                        timestep + self.policy.curr_motion_plan_steps,
                        timestep
                        + self.policy.curr_motion_plan_steps
                        + self.policy.curr_constraint_steps,
                    )
                )
                constraint_sequence.append(curr_constraint)

            action_to_take = np.array(action)

            # goal: simulate errors that robot might see during test time
            actions_left = len(self.policy.action_queue)
            n_constraint_segment_actions = self.policy.curr_constraint_steps
            # 判断：如果剩余动作数 < 约束步数，说明已经走完了赶路阶段，正在干精细活（如抓取）。
            in_constraint_segment = actions_left < n_constraint_segment_actions
            # # 判断：是不是最后一个阶段（通常是放置）。
            is_last_constraint_segment = (
                self.policy.current_constraint_idx
                == len(self.policy.current_constraint_sequence) - 1
            )
            # 根据阶段施加不同强度的噪声。
            if in_constraint_segment and not is_last_constraint_segment:
                # 如果是中间的干活阶段（如抓取），加 Constraint 级别的噪声（通常较小，怕抖掉了）。
                action_to_take += np.random.uniform(-1, 1, action.shape) * np.array(
                    system_noise_cfg.constraint_segment_noise_magnitude
                )
            elif not in_constraint_segment:
                # 如果是赶路阶段，加 Motion 级别的噪声（通常较大，增加轨迹多样性）。
                action_to_take += np.random.uniform(-1, 1, action.shape) * np.array(
                    system_noise_cfg.motion_segment_noise_magnitude
                )
            # mightn't be the best great to add noise in constraint segment --- however, needed for stuff like grasping?
            # another strategy is to make the guy 'see' incorrect poses and try to solve for the constraint preservation
            # in this way it's more correlated but still centered?
            # issue: still want the 'correct' actions to train on ... we'll want to ignore the add-noise script in this case
            # issue with the above approach is it may lead to bias on the failure cases ...

            # 执行动作，并记录数据
            obs, reward, done, info = self.env.step(action_to_take)
            timestep += 1
            obss.append(obs)
            state = self.env.get_state()
            states.append(state["states"])
            actions.append(action)
            rewards.append(reward)

            if self.policy.check_constraint and gen_single_stage_only:
                # 如果开启了“只生成单阶段”模式，且当前阶段刚刚结束（check_constraint 为真）
                if is_env_state_close(self.env, self.policy.current_constraint):
                    # # 检查物理状态是否真的满足了约束（比如真的抓住了）。
                    #             # 如果满足，强制把最后一步奖励设为 1（成功），并提前结束。
                    # TODO(klin): check if state is close enough to original state ...
                    # check constraint means flipped to new constraint
                    rewards[-1] = 1  # hardcode success to keep single stage
                break
            if done:
                logging.info("Episode ended before success.")
                # currently unused, but need rewards
                break

            if self.policy.check_constraint:
                # 如果一个阶段刚结束
                # TODO: move this out of demo generator loop if possible?
                # 收集刚才生成的所有图片
                obs_lst: List[np.ndarray] = [
                    obs_dct.get("agentview_image", None) for obs_dct in obss
                ]
                if self.policy.verbose:
                    # 如果开启详细模式，弹窗显示当前的机器人构型。
                    # visualize the robot configurations
                    create_passive_viewer(
                        self.policy.robot_configuration.model,
                        self.policy.robot_configuration.data,
                        show_left_ui=False,
                        show_right_ui=False,
                    )
                # TODO: fix the obs please
                # # 保存视频片段，方便人工检查这一段跑得对不对。
                if obs_lst[0] is not None:
                    save_images(
                        obs_lst, "datasets/generated/images-new-full/", save_as_mp4=True
                    )
        self.policy.reset()

        if len(actions) == 0:
            # 如果因为某种严重错误（比如第一步规划就崩了），导致 actions 列表是空的。
            # 为了防止后续保存 HDF5 时因为数组为空而报错（Crash），强行塞一个全 0 的假动作进去。
            # 让程序“活着”把错误记录下来，而不是直接崩溃。
            logging.warning(
                "Using dummy action for failed demo. TODO: reproduce failure for diagnosis."
            )
            actions.append(np.zeros(7))

        success = (rewards[-1] == 1) if rewards else False

        # quirk of mujoco based envs that directly model the mjmodel
        # 把“物体被焊在手上”这一物理拓扑结构的改变，固化到 XML 字符串中保存下来。
        if isinstance(self.env, CPEnvRobomimic):
            model = update_fixed_joint_objects_in_xml(self.env.env.env.sim.model, model)

        if store_single_stage_only and len(constraint_sequence) > 0:
            assert (
                store_single_stage_max_extra_steps >= 0
            ), "store_single_stage_max_extra_steps must be non-negative"
            # 根据max_steps对数据进行截断
            constraint_last_tstep = constraint_sequence[0].timesteps[-1]
            max_steps = min(
                constraint_last_tstep + store_single_stage_max_extra_steps, len(obss)
            )
            obss = obss[:max_steps]
            actions = actions[:max_steps]
            dones = dones[:max_steps]
            rewards = rewards[:max_steps]
            infos = infos[:max_steps]
            states = states[:max_steps]
            constraint_sequence = [constraint_sequence[0]]

        # 把里面的绝对路径替换成相对路径或占位符
        if self.anonymize_model_file_paths:
            model = anonymize_model_file_paths(model)

        # Save the episode states/actions
        episode_data: Dict[str, Any] = {
            "observations": obss,
            "actions": np.array(actions),
            "dones": dones,
            "rewards": rewards,
            "infos": infos,
            "states": np.array(states)[: len(actions)],
            "model_file": model,  # temp: use original model file to avoid get_xml() issues
            "success": success,
            "failure_type": failure_type,
            "constraint_sequence": constraint_sequence,
        }
        return episode_data


def save_demo(demo: Dict[str, Any], save_path: str, env_meta: str):
    # Open the HDF5 file in write mode
    with h5py.File(save_path, "w") as f:
        # Create a group for the demo
        # create_group类似创建文件夹
        demo_group = f.create_group("data/demo_0")
        # Save each dataset in the demo group
        # create_dataset类似创建文件
        demo_group.create_dataset("actions", data=np.array(demo["actions"]))
        demo_group.create_dataset("states", data=np.array(demo["states"]))
        demo_group.attrs["success"] = (
            demo["success"] if demo["success"] is not None else False
        )
        # Save failure type
        demo_group.attrs["failure_type"] = (
            demo["failure_type"] if demo["failure_type"] is not None else "None"
        )
        if demo["constraint_sequence"]:
            cgrp = demo_group.create_group("constraint_data")
            for i, c in enumerate(demo["constraint_sequence"]):
                cdict = c.to_constraint_data_dict()
                cgrp.create_dataset(f"constraint_{i}", data=json.dumps(cdict))

        demo_group.attrs["model_file"] = demo["model_file"]
        f["data"].attrs["env_args"] = json.dumps(env_meta)

    logging.info(f"Saved demo to {save_path}")


def save_images(
    images: List[np.ndarray], folder_path: str, save_as_mp4: bool = False, fps: int = 20
):
    if len(images) == 0 or images[0] is None:
        logging.info("No images to save.")
        return
    folder_path = pathlib.Path(folder_path)
    folder_path.mkdir(parents=True, exist_ok=True)

    if save_as_mp4:
        output_path = folder_path / "video.mp4"
        # Convert images from CHW to HWC and scale to [0, 255] range
        processed_images = [
            (np.transpose(image, (1, 2, 0)) * 255).astype(np.uint8) for image in images
        ]

        # Save images as an MP4 video
        imageio.mimwrite(str(output_path), processed_images, format="mp4", fps=fps)
    for i, image in enumerate(images):
        image_path = folder_path / f"image_{i}.png"
        # Convert from CHW to HWC
        image = np.transpose(image, (1, 2, 0))
        # Convert from [0, 1] to [0, 255]
        image = (image * 255).astype(np.uint8)
        imageio.imwrite(image_path, image)


def run_command(command: str) -> str:
    """Run a shell command and return its output as a string."""
    # shell=True:通过系统 Shell（如 /bin/sh 或 cmd.exe）来运行这条命令.False时只能运行可执行文件
    # check=True:如果命令执行失败（报错），请立刻让 Python 崩溃（抛出异常）。
    # capture_output=True:把命令的标准输出（stdout）和错误输出（stderr）捕获到 result 对象里
    # text=True:把输出结果自动解码成字符串（String）(默认情况下，subprocess 捕获的是 字节流（bytes, b'...'）)
    result = subprocess.run(
        command, shell=True, check=True, capture_output=True, text=True
    )
    return result.stdout


def extract_git_installed_packages(requirements_content: str) -> list:
    """Extract Git-installed packages from the requirements content."""
    git_packages = []
    for line in requirements_content.splitlines():
        if re.match(r"^-e git+|git+|http", line):
            match = re.search(r"#egg=([^\s]+)", line)
            if match:
                git_packages.append(match.group(1))
    return git_packages


def export_conda_environment(
    env_name: str, exclude_packages: list, env_yaml_path: str
) -> str:
    """Export Conda environment to environment.yaml excluding specified packages."""
    conda_export_command = "conda env export --no-builds"
    conda_env_content = run_command(conda_export_command)
    with open(env_yaml_path, "w") as yaml_file:
        pip_section = False
        for line in conda_env_content.splitlines():
            # Skip the prefix line for portability
            # 原始输出通常包含一行 prefix: /home/username/anaconda3/envs/myenv，将其过滤，避免隐私泄露
            if line.startswith("prefix:"):
                continue

            # Detect pip section start
            if "pip:" in line:
                pip_section = True
                yaml_file.write(line + "\n")
                continue

            # Handle pip packages
            # 过滤exclude_packages
            if pip_section:
                line_split = line.split()
                package_split = line_split[1].split("==")
                # Check if line contains an excluded package
                package_name = package_split[0]
                if any(package_name == pkg for pkg in exclude_packages):
                    logging.info(f"Excluding package: {package_name}")
                    continue

            yaml_file.write(line + "\n")
    logging.info(
        "Conda environment exported as environment.yaml (Git-installed pip packages excluded)"
    )
    return env_yaml_path


def save_to_wandb(file_path: str, artifact_name: str, artifact_type: str):
    """Save a file to wandb as an artifact."""
    # “Artifact” 指的是实验过程中产生的任何文件：数据集、模型权重、配置文件、代码快照等。
    # WandB 通过 Artifact 系统来对这些文件进行版本控制。
    # artifact_type: 文件的类别（例如 model, dataset, code）。
    artifact = wandb.Artifact(name=artifact_name, type=artifact_type)
    artifact.add_file(file_path)  # 这一步并没有上传文件，只是建立了本地文件和 Artifact 对象之间的引用关系。
    wandb.log_artifact(artifact)  # 把 Artifact 对象（以及它引用的那个文件）上传到 WandB 的云端服务器
    logging.info(f"Saved {file_path} to WandB as {artifact_name}")


@dataclass
class LoggingConfig:
    logging_path: Optional[str] = None

    def setup_logging(self):
        if self.logging_path is None:
            # add datetime for unique logging
            self.logging_path = (
                pathlib.Path("logs")
                / datetime.now().strftime("%Y%m%d-%H%M%S-%f")
                / "demo-gen.log"
            )

        logging.getLogger("curobo").setLevel(logging.WARNING)
        pathlib.Path(self.logging_path).parent.mkdir(parents=True, exist_ok=True)
        setup_file_logger(self.logging_path, backup_count=1)
        print(f"Logging to {self.logging_path}")


@dataclass
class InitializationConfig:
    initialization_noise_type: str = "gaussian"
    initialization_noise_magnitude: float = 0.02


@dataclass
class Config:
    seed: int = 0
    demo_path: str = "datasets/source/square-demo-152.hdf5"  # 源演示路径
    download_demo: bool = False  # 如果本地没有这个文件，是否去 HuggingFace 下载。
    load_demos_start_idx: int = 0
    load_demos_end_idx: int = 1
    debug: bool = False
    ignore_obj_geom_scaling: bool = False # 如果设为 True，计算动作时会忽略物体尺寸的变化比例。
    """Whether to ignore object geometry scaling during demo generation. Set to true to approximate mimicgen."""
    use_reset_near_constraint: bool = False  # 是否启用“随机重置到约束附近”的功能。如果 True，机器人不会从头开始，而是瞬移到任务的关键节点（如抓取前）。
    use_reset_to_state: bool = False  # 是否强制重置到某个特定的历史物理状态。
    gen_single_stage_only: bool = False  # 只生成单阶段：如果为 True，只要完成了当前的子任务（如抓取成功），就立刻停止录制。
    store_single_stage_only: bool = False  # 只保存单阶段：如果为 True，保存数据时只切取当前阶段的帧，丢弃之前的帧。
    """Whether to only store a single stage of a demo; used for reset-near-constraint balancing"""
    store_single_stage_max_extra_steps: int = 16  # 多录一点：在阶段结束（Success）后，额外再多录 16 步。目的：为了让模仿学习算法能学到未来的动作趋势。
    """Since we predict a sequence of actions, helpful to store extra actions after end of stage"""
    system_noise_cfg: SystemNoiseConfig = field(
        default_factory=SystemNoiseConfig
    )  # 定义了在运动规划和执行过程中，给动作加入多大的随机抖动
    """System noise configuration for motion planning and constraint execution for plausible observation diversity"""
    # 配合 use_reset_to_state 使用，指定具体去读取哪个文件里的第几条演示作为初始状态。
    reset_state_demo_path: Optional[str] = None
    reset_state_demo_idx: Optional[int] = None
    # eef_interp_curobo:先试着走直线，如果行不通，再切换至curobo
    motion_planner_type: Literal[
        "eef_interp", "curobo", "eef_interp_mink", "eef_interp_curobo"
    ] = "eef_interp_curobo"
    curobo_goal_type: Literal["joint", "pose", "pose_wxyz_xyz"] = "pose_wxyz_xyz"
    robot_type: Literal["franka", "franka_umi"] = "franka"
    n_demos: int = 1  # 生成的demo数量
    require_n_demos: bool = False  # True：必须生成 n_demos 条成功的数据才停。False：跑 n_demos 次循环就停，不管成不成功。
    motion_plan_save_dir: Optional[str] = None  # 运动规划结果保存路径
    constraint_selection_method: Literal[
        "random", "first", "second", "third", "last"
    ] = "random"  # 当穿越模式下，随机穿越到哪一关？（随机、第一关、最后一关等）。
    custom_constraints_path: Optional[str] = None  # 自定义约束文件：如果不想用自动生成的约束，可以自己提供一个 JSON 文件。
    """Custom path to constraints (of a demo) file"""
    override_constraints: bool = False  # 强制重算约束：即使存在缓存文件，也强制重新计算几何约束。
    """Whether to override old constraints with new ones"""
    override_interactions: bool = False  # 强制重标交互：强制重新询问用户输入交互信息（Hand:Object）。
    """Whether to override old interactions with new ones"""
    interaction_threshold: float = 0.03  # 交互阈值：手和物体距离小于 3cm 算作接触。
    """Threshold for interaction detection"""
    demo_segmentation_type: Optional[Literal["distance-based", "llm-e2e", "llm-success"]] = None
    controller_type: Literal["ik", "default"] = "ik"
    logging: LoggingConfig = field(default_factory=LoggingConfig)
    save_dir: Optional[pathlib.Path] = None  # 总保存目录：生成的 HDF5、视频、日志都放在这。
    merge_demo_save_path: Optional[str] = None  # 合并文件路径：最后打包的大 HDF5 文件存哪里。
    save_motion_plans: bool = True  # 是否保存运动规划的详细轨迹
    use_wandb: bool = False  # 是否启动wandb
    wandb_project: Optional[str] = "cpgen"  # 项目名称
    wandb_entity: Optional[str] = None  # WandB 用户名或者团队名
    wandb_name_prefix: Optional[str] = None  # 实验命名前缀
    wandb_run_url: Optional[str] = None  # 本次实验的网页链接
    env_name: Optional[str] = None
    initialization: InitializationConfig = field(default_factory=InitializationConfig)  # 环境重置时的随机扰动配置

    def __post_init__(self):
        if self.motion_plan_save_dir:
            self.motion_plan_save_dir = pathlib.Path(self.motion_plan_save_dir)
        
        if not self.merge_demo_save_path:
            self.merge_demo_save_path = f"datasets/generated/{self.env_name}/merged_demos.hdf5"

        self.validate_motion_planner_configs()
        self.validate_reset_to_state_configs()
        if self.download_demo:
            self._handle_demo_download()
        elif not pathlib.Path(self.demo_path).exists():
            self._prompt_demo_download()

        self.set_env_name()
        self.set_save_directory()
        self.setup_logging()
        self.setup_motion_plan_directory()
        self.setup_wandb_logging()

    def setup_logging(self):
        self.logging.logging_path = self.save_dir / "demo-gen.log"
        self.logging.setup_logging()

    def set_env_name(self):
        if self.env_name is not None:
            return
        import robomimic.utils.file_utils as FileUtils

        env_meta = FileUtils.get_env_metadata_from_dataset(dataset_path=self.demo_path)
        env_name = env_meta.get("env_name", "unknown_env")
        self.env_name = env_name

    def setup_wandb_logging(self):
        if not self.use_wandb:
            logging.info("WandB logging disabled.")
            return

        import wandb

        def wandb_thread():
            wandb.init(
                project=self.wandb_project,
                entity=self.wandb_entity,
                config=asdict(self),
                name=f"{self.wandb_name_prefix}_{self.env_name}_{datetime.now().strftime('%Y%m%d-%H%M%S')}",
            )
            logging.info("WandB logging initialized.")

            self.wandb_run_url = wandb.run.url
            print(f"WandB run ID: {self.wandb_run_url}")

            # Export environment and log to WandB
            logging.info("Identifying Git-installed pip packages...")
            requirements_content = run_command("pip freeze")
            requirements_path = f"{self.save_dir}/requirements.txt"
            with open(requirements_path, "w") as req_file:
                req_file.write(requirements_content)

            git_packages = extract_git_installed_packages(requirements_content)

            logging.info(
                "Exporting Conda environment to environment.yaml (excluding Git-installed pip packages)..."
            )
            assert self.env_name is not None, "Environment name must be set."
            env_yaml_path = export_conda_environment(
                env_name=self.env_name,
                exclude_packages=git_packages,
                env_yaml_path=f"{self.save_dir}/environment.yaml",
            )

            logging.info("Logging generated files to WandB...")
            save_to_wandb(requirements_path, "requirements", "pip-requirements")
            save_to_wandb(env_yaml_path, "conda-environment", "environment")

            logging.info("Export completed. Files generated:")
            logging.info(
                "  - environment.yaml (Conda environment, excluding Git-installed pip packages)"
            )
            logging.info(
                "  - requirements.txt (all pip packages, including Git-installed packages)"
            )

        thread = threading.Thread(target=wandb_thread)
        thread.start()

    def validate_motion_planner_configs(self):
        """检查设置的运动规划器参数是否存在逻辑冲突"""
        if self.motion_planner_type == "eef_interp_curobo":
            assert self.curobo_goal_type == "pose_wxyz_xyz", (
                "Curobo motion planner requires goal type 'pose_wxyz_xyz' for now. "
                "Please update motion planner configs."
            )
        if self.motion_planner_type == "curobo":
            assert (
                self.curobo_goal_type == "pose_wxyz_xyz"
                or self.curobo_goal_type == "joint"
            ), (
                "Curobo motion planner requires goal type 'pose_wxyz_xyz' or 'joint' for now. "
                "Please update motion planner configs."
            )

    def validate_reset_to_state_configs(self):
        if self.use_reset_to_state:
            # 如果开关开了，那么 reset_state_demo_path（存档文件路径）和 reset_state_demo_idx（存档编号）都不能是空的（None）。
            assert (
                self.reset_state_demo_path is not None
                and self.reset_state_demo_idx is not None
            ), "If 'use_reset_to_state' is True, 'reset_state_demo_path' and 'reset_state_demo_idx' must be provided."
        if self.reset_state_demo_path is not None:
            # 检查文件是否存在
            assert (
                pathlib.Path(self.reset_state_demo_path).exists()
            ), f"Reset state demo file not found at {self.reset_state_demo_path}."

    def set_save_directory(self):
        if self.save_dir is None:
            self.save_dir = pathlib.Path(
                f"datasets/generated/{self.env_name}/{datetime.now().strftime('%Y-%m-%d-%H-%M-%S')}"
            )
        (self.save_dir / "successes").mkdir(parents=True, exist_ok=True)
        (self.save_dir / "failures").mkdir(parents=True, exist_ok=True)

    def setup_motion_plan_directory(self):
        if not self.save_motion_plans:
            self.motion_plan_save_dir = None
            logging.warning(
                "Motion planning saving is disabled. No plans will be saved."
            )
        elif self.motion_plan_save_dir is None:
            self.motion_plan_save_dir = self.save_dir / "motion_plans"
            self.motion_plan_save_dir.mkdir(parents=True, exist_ok=True)

    def _handle_demo_download(self):
        """Handle the downloading of demo datasets from HuggingFace Hub."""
        # Define mapping of local filenames to HF repo files
        repo_id = "cpgen/datasets-src-private"
        self._download_from_hub(repo_id, self.demo_path)

    def _download_from_hub(self, repo_id: str, filename: str):
        """Download a file from HuggingFace Hub."""
        from huggingface_hub import hf_hub_download

        try:
            # Create parent directories if they don't exist
            pathlib.Path(self.demo_path).parent.mkdir(parents=True, exist_ok=True)

            TOKEN = os.getenv("HUGGING_FACE_HUB_TOKEN")
            if TOKEN is None:
                TOKEN = input("Please enter your Hugging Face Hub token: ")

            # Download file from HuggingFace Hub
            downloaded_path = hf_hub_download(
                repo_id=repo_id,  # 仓库名
                filename=filename,  # 文件名
                local_dir=pathlib.Path(demo_aug.__file__).parent.parent.as_posix(),  # 制定下载的根目录
                repo_type="dataset",  # 声明这是一个数据集仓库（而不是模型仓库）
                token=TOKEN,
            )

            # If the download location differs from our target path, move the file
            downloaded_path = pathlib.Path(downloaded_path)
            if downloaded_path != self.demo_path:
                # 如果不一致，执行重命名操作（相当于移动文件）。
                # 把文件从下载缓存区移到 self.demo_path 指定的位置。
                downloaded_path.rename(self.demo_path)

            print(f"Successfully downloaded {filename} to {self.demo_path}")

        except Exception as e:
            logging.error(f"Error downloading file from HuggingFace Hub: {e}")
            if pathlib.Path(self.demo_path).exists():
                pathlib.Path(self.demo_path).unlink()  # 删除文件
            raise

    def _prompt_demo_download(self):
        """Prompt user to download demo if path doesn't exist."""
        # 当程序发现硬盘上没有指定的演示文件时，不直接报错退出，而是暂停下来问用户：“文件没找到，要不要我现在帮你从网上下载一个？”
        print(f"Demo file not found at {self.demo_path}")
        response = input("Would you like to download the demo file? (y/n): ").lower()

        if response == "y":
            logging.info(
                f"Downloading demo file {self.demo_path} from huggingface. Please ensure you're authenticated."
            )
            self.download_demo = True
            self._handle_demo_download()
        else:
            print("Please ensure the demo file is available at the specified path.")


def get_eef_configuration(
    model_xml: str,
    mj_model: mujoco.MjModel,
    mj_data: mujoco.MjData,
    return_model_only: bool = False,
    gripper_joint_names: List[str] = None,
) -> Union[IndexedConfiguration, mujoco.MjModel]:
    """
    Generates the end-effector configuration from the given model XML (which includes the full robot and environment).
    We have hardcoded the logic for detaching the eef from the arm and correctly adding a free joint to the eef.

    Args:
        model_xml (str): The XML string representing the model.
        return_model_only (bool, optional): If True, only the MjModel object is returned.
            If False (default), a Configuration object is returned.
        gripper_joint_names (List[str], optional): The joint names of the gripper.
    """
    xml = update_xml_with_mjmodel(
        model_xml,
        mj_model,
        exclude_body_and_children=["gripper0_right_right_gripper"],
    )
    # 保留 gripper0_right_right_gripper（夹爪），删除 robot0_base（基座）以及两者之间的连接（手臂）。
    xml = remove_arm_keep_eef(xml, "robot0_base", "gripper0_right_right_gripper")
    # 删除对应的电机驱动
    xml = remove_actuator_tag(
        xml
    )  # because may have removed certains arm joints that actuators correspond to
    # 原本手是连在手臂上的，手臂动，手才动。
    # 现在手独立了，为了让它能在空间中移动（x, y, z）和旋转（roll, pitch, yaw），必须给它加一个 6自由度关节（Free Joint）。
    xml = add_free_joint(
        xml, "gripper0_right_right_gripper", "right_free_joint"
    )  # add a free joint to the eef
    # 把改好的6自由度的“夹爪模型”保存到文件，方便调试看效果。
    with open("eef_and_env_model.xml", "w") as f:
        f.write(xml)
    eef_and_env_model = mujoco.MjModel.from_xml_string(xml)
    if return_model_only:
        return eef_and_env_model
    # 获取新模型中所有关节的索引映射
    joint_name_to_idxs = get_joint_name_to_indexes(eef_and_env_model)
    # 定义这个“新机器人”的所有可控关节：
    #     Free Joint：控制手在空中的位置和姿态。
    #     Gripper Joints：控制手指的开合。
    all_robot_joints = ["right_free_joint"] + gripper_joint_names
    # 获取可控关节对应的id
    robot_idxs = np.concatenate(
        [joint_name_to_idxs[joint_name] for joint_name in all_robot_joints]
    )
    # 创建一个封装对象。它不仅包含模型，还记住了6自由度夹爪的关节数据在整个物理状态数组中的具体位置（Index）。
    indexed_configuration = IndexedConfiguration(
        eef_and_env_model, robot_idxs=robot_idxs
    )
    # 使用原来的模型和数据对新模型的对应部分进行初始化
    sync_mjdata(
        mj_model, mj_data, indexed_configuration.model, indexed_configuration.data
    )
    return indexed_configuration


def get_robot_configuration(
    model_xml: str,
    mj_model: mujoco.MjModel,
    mj_data: mujoco.MjData,
    return_model_only: bool = False,
    robot_joint_names: List[str] = None,
) -> Union[IndexedConfiguration, mujoco.MjModel]:
    """
    从一个包含完整环境和机器人的 XML 模型中，构建出一个专门用于几何计算的机器人配置对象
    1. 它拿到了原始环境的 XML 和当前状态。
    2. 它修剪了 XML（移除驱动器，排除基座）。
    3. 它生成了一个新的 MuJoCo 模型。
    4. 它把当前时刻的所有数据同步进去。
    5. 最终返回一个方便操作的配置对象。
    Generates the robot configuration from the given model XML (which includes the full robot and environment).
    We have hardcoded the logic for detaching the eef from the arm and correctly adding a free joint to the eef.

    Args:
        model_xml (str): The XML string representing the model.
        return_model_only (bool, optional): If True, only the MjModel object is returned.
            If False (default), a Configuration object is returned.

    Returns:
        Union[Configuration, mujoco.MjModel]: The robot configuration or the MjModel object.
    """
    # 把当前的 mj_model 的状态（比如物体位置）写回到 model_xml 文本
    # 特意排除了 "robot0_link0"（基座连接件），基座通常是焊死在地面上的，相对位置永远不变。如果强制更新它，可能会因为浮点数误差导致模型“微小的抖动”或错位，所以直接保留原始 XML 里的定义最安全。
    xml = update_xml_with_mjmodel(
        model_xml,
        mj_model,
        exclude_body_and_children=[
            "robot0_link0"
        ],  # link welded to base, avoid updating their relative pose
    )
    # 移除动力学驱动-电机
    #新生成的模型主要用于几何计算（比如算距离、算碰撞），不需要电机（Actuators）来驱动它动起来。而且，之前的步骤可能修改了关节结构，如果保留 Actuator 标签，可能会因为找不到对应的关节而报错。
    xml = remove_actuator_tag(
        xml
    )  # because may have removed certains arm joints that actuators correspond to
    robot_env_model = mujoco.MjModel.from_xml_string(xml)
    if return_model_only:
        return robot_env_model
    # need robot_idxs: need joint names: acquire from original env maybe ...
    # 得到一个映射：关节名->id
    robot_joint_name_to_idxs = get_joint_name_to_indexes(robot_env_model)
    # 根据传入的 robot_joint_names（比如 ["joint1", "joint2"...]），查出它们在新模型中对应的数字索引（Indices）
    # 为了后续能快速修改这些关节的角度
    robot_idxs = np.concatenate(
        [robot_joint_name_to_idxs[joint_name] for joint_name in robot_joint_names]
    )
    # 创建一个 IndexedConfiguration 对象。
    # 这是一个自定义类，它把模型（Model）、数据（Data）和关节索引（robot_idxs）打包在一起，方便管理。
    indexed_configuration = IndexedConfiguration(robot_env_model, robot_idxs=robot_idxs)
    # 把原始环境 (mj_model, mj_data) 的当前状态（所有关节角度、速度、物体位置），完整复制到这个新创建的影分身 (indexed_configuration.data) 中
    sync_mjdata(
        mj_model, mj_data, indexed_configuration.model, indexed_configuration.data
    )
    return indexed_configuration


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
            render=False,
        )
    )
    # 跑得快，纯物理计算
    src_env = EnvUtils.create_env_from_metadata(  # not great that we need to specify these keys
        env_meta=src_env_meta,
        use_image_obs=False,
        render_offscreen=False,
        render=False,
    )
    src_env = CPEnvRobomimic(src_env)
    env_meta["env_name"] = cfg.env_name
    env_meta["env_version"] = robosuite.__version__
    # env是交互环境，用来跑新的仿真。
    env = EnvUtils.create_env_from_metadata(  # not great that we need to specify these keys
        env_meta=env_meta,
        use_image_obs=False,
        render_offscreen=False,
        render=False,
    )
    env = CPEnvRobomimic(env)
    # 分析原始的demo，并提取机器人可以理解的几何约束
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
    src_env_w_rendering.env.close()  # so that rendering works properly
    # 主规划器，用于避障和长距离规划
    if cfg.motion_planner_type == "eef_interp":
        # 末端插值规划器：走直线
        motion_planner = EEFInterpMotionPlanner(
            env.env, save_dir=cfg.motion_plan_save_dir
        )
    elif cfg.motion_planner_type == "curobo":
        # CuRobo 规划器：CuRobo 是 NVIDIA 开发的一个基于 CUDA 加速的机器人库
        # 它使用轨迹优化算法（Trajectory Optimization）。它可以生成平滑的、能够自动避开障碍物的曲线路径。
        motion_planner = CuroboMotionPlanner(
            env.env,
            save_dir=cfg.motion_plan_save_dir,
            goal_type=cfg.curobo_goal_type,
            robot_type=cfg.robot_type,
        )
    elif cfg.motion_planner_type == "eef_interp_mink":
        # Mink 插值规划器：末端依然走直线，但是使用Mink来解算IK
        motion_planner = EEFInterpMinkMotionPlanner(
            env.env, save_dir=cfg.motion_plan_save_dir
        )
    elif cfg.motion_planner_type == "eef_interp_curobo":
        # 混合规划器：当你确定两点之间没有障碍物，想要走最短路径，但又希望利用 CuRobo 极快的求解速度和极高的求解精度时使用。
        motion_planner = EEFInterpCuroboMotionPlanner(
            env.env,
            save_dir=cfg.motion_plan_save_dir,
            mink_robot_configuration=Configuration(
                env.env.env.robots[0].robot_model.mujoco_model,
            ),
            curobo_goal_type=cfg.curobo_goal_type,
            robot_type=cfg.robot_type,
        )
    else:
        raise ValueError(f"Invalid motion planner type: {cfg.motion_planner_type}")

    # 辅助规划器，专门用于处理微小调整或严格直线运动
    eef_interp_mink_motion_planner = EEFInterpMinkMotionPlanner(env.env)

    original_xml = env.env.env.sim.model.get_xml()
    # 把 XML 图纸和当前的物理数据结合，创建了一个描述整个机器人（手臂 + 夹爪）的对象
    robot_configuration = get_robot_configuration(
        original_xml,
        env.env.env.sim.model._model,
        env.env.env.sim.data._data,
        robot_joint_names=env.env.env.robots[0].robot_model.joints
        + env.env.env.robots[0].robot_model.grippers["robot0_right_hand"].joints,
    )
    # 提取**夹爪（End Effector）**的信息
    # 策略需要单独知道夹爪的结构，以便精确控制抓取动作
    eef_configuration = get_eef_configuration(
        original_xml,
        env.env.env.sim.model._model,
        env.env.env.sim.data._data,
        gripper_joint_names=env.env.env.robots[0]
        .robot_model.grippers["robot0_right_hand"]
        .joints,
    )
    # 工作流程是：
    #     读取 constraint_sequences 里的下一个约束（比如“手要在杯子上 5cm”）。
    #     结合 robot_configuration 知道当前手在哪。
    #     使用 motion_planner 或 eef_interp_mink_motion_planner 计算一条路径，从“当前位置”移动到“满足约束的位置”。
    #     输出动作给机器人执行。
    cp_policy = CPPolicy(
        constraint_sequences,  # 1. 之前生成的几何约束序列
        motion_planner,  # 2. 主运动规划器
        robot_configuration,  # 3. 自身状态：机器人的全身体检报告
        eef_configuration,  # 4. 抓手状态：夹爪的详细报告
        eef_interp_mink_motion_planner,  # 5. 精细工具：刚才初始化的 Mink IK 求解器
        viz_robot_kpts,  # 6. 可视化数据：用于在界面上画出关键点的辅助数据
        env,  # 7. 环境句柄：用于和环境交互
        original_xml=original_xml,  # 8. 场景描述：XML 原文件
        ignore_scaling=cfg.ignore_obj_geom_scaling  # 9. 配置标志
    )

    # get a motion planning model where I can attach objects to the env if needed
    demo_generator = DemoGenerator(
        env,
        cp_policy,
        use_reset_near_constraint=cfg.use_reset_near_constraint,  # 是否开启随机穿越模式，如果是 False：机器人每次都老老实实从 t=0（初始位置）开始做任务。如果是 True：机器人会穿越到任务的中途。
        constraint_selection_method=cfg.constraint_selection_method,  # "random"：随机选一关（Constraint）。一会儿练抓取，一会儿练放置。"first"：总是穿越到第一关。"last"：总是穿越到最后一关。
        use_reset_to_state=cfg.use_reset_to_state,  # 如果为 True，机器人不会随机初始化，而是完全克隆硬盘上某个 HDF5 文件里的某一帧状态。
        reset_state_demo_path=cfg.reset_state_demo_path,
        reset_state_demo_idx=cfg.reset_state_demo_idx,
        demo_src_env=src_env,  # currently used for resetting to state; mightn't need after getting poses from demos
    )

    trials = 0
    n_successes = 0
    demos = []
    success_demo_save_paths = []
    fail_demo_save_paths = []
    keep_generating = True  # 循环开关：控制是否继续生成（比如达到 N 条成功数据后设为 False）

    # 告知日志位置 (Log Location)
    for handler in logging.getLogger().handlers:
        if isinstance(handler, logging.FileHandler):
            print("Data generation logging in log file path:", handler.baseFilename)

    while keep_generating:
        demo = demo_generator.generate_demo(
            # True: 机器人只要完成了当前这一个约束阶段（比如“抓取成功”），程序就立刻判定为 Success 并结束录制。
            # False (默认): 机器人会尝试跑完整个长任务。例如：伸手 -> 抓取 -> 移动 -> 放置。直到任务彻底完成才结束。
            gen_single_stage_only=cfg.gen_single_stage_only,
            # False (默认): 保存从 t=0 开始到结束的所有数据。
            # True: 即使机器人跑了很久，保存数据时，我只截取当前关注的这个阶段（加上后面的一小段缓冲）。
            store_single_stage_only=cfg.store_single_stage_only,
            # 如果 store_single_stage_only=True，当阶段结束（比如刚抓到物体）时，会再多录制几步。
            # 现代模仿学习模型（如 ACT, Diffusion Policy）通常会预测未来的动作序列（Chunking）。
            # 如果数据在抓到物体的一瞬间戛然而止，模型就学不到“抓到之后该往哪动”的趋势。多录这 16 步就是为了给模型提供Future Horizon。
            store_single_stage_max_extra_steps=cfg.store_single_stage_max_extra_steps,
            # 定义了在“移动阶段”和“抓取阶段”分别要给动作加上多大的随机噪声（高斯噪声/均匀噪声）。
            system_noise_cfg=cfg.system_noise_cfg,
        )
        intermediate_folder = "successes" if demo["success"] else "failures"
        save_path = (
            cfg.save_dir
            / intermediate_folder
            / (datetime.now().strftime("%H-%M-%S-%f") + ".hdf5")
        )

        save_demo(demo, save_path.as_posix(), env_meta)
        demos.append(demo)
        if demo["success"]:
            success_demo_save_paths.append(save_path)
        else:
            fail_demo_save_paths.append(save_path)

        trials += 1
        n_successes += int(demo["success"])
        # 质量优先。“我不凑够 N 个成功的就不下班！”
        if cfg.require_n_demos and n_successes == cfg.n_demos:
            break
        # 数量优先。“我就给你 N 次机会，跑完拉倒。”
        if not cfg.require_n_demos and trials == cfg.n_demos:
            break

        logging.info(f"Trials: {trials}, Successes: {n_successes}")

    logging.info(f"Trials: {trials}, Successes: {n_successes}")

    from demo_aug.utils.robomimic_utils import playback_trajectory_with_env

    env_meta["env_kwargs"]["use_camera_obs"] = False
    env_meta["env_kwargs"]["has_offscreen_renderer"] = True
    env = EnvUtils.create_env_from_metadata(  # not great that we need to specify these keys
        env_meta=env_meta,
        use_image_obs=env_meta["env_kwargs"]["use_camera_obs"],
        render_offscreen=env_meta["env_kwargs"]["has_offscreen_renderer"],
        render=not env_meta["env_kwargs"]["has_offscreen_renderer"],
    )
    merge_demo_save_path = cfg.merge_demo_save_path
    if success_demo_save_paths:
        save_path = pathlib.Path(success_demo_save_paths[0].parent) / (
            pathlib.Path(success_demo_save_paths[0]).stem + ".hdf5"
        )
        total_demos = count_total_demos(success_demo_save_paths)
        if merge_demo_save_path is None:
            merge_demo_save_path = pathlib.Path(save_path).parent / (
                str(pathlib.Path(save_path).stem)
                + f"_{total_demos}demos"
                + str(pathlib.Path(save_path).suffix)
            )
        merge_demo_files(success_demo_save_paths, save_path=merge_demo_save_path)

    if fail_demo_save_paths:
        merge_failure_demo_save_path = pathlib.Path(merge_demo_save_path).parent / (
            str(pathlib.Path(merge_demo_save_path).stem)
            + "_failures"
            + str(pathlib.Path(merge_demo_save_path).suffix)
        )
        merge_demo_files(fail_demo_save_paths, save_path=merge_failure_demo_save_path)
        # always save failure videos
        for i, demo in enumerate(demos):
            if demo["success"]:
                continue

            initial_state = {
                "model": demo["model_file"],
                "states": demo["states"][0],
            }

            intermediate_folder = "failures"
            save_video_path = (
                cfg.save_dir
                / intermediate_folder
                / f"cpgen-mp={cfg.motion_planner_type}-seed={cfg.seed}-demo={i}.mp4"
            )

            video_writer = imageio.get_writer(save_video_path, fps=20)
            # 生成了一个包含多视角的视频
            playback_trajectory_with_env(
                env,
                initial_state=initial_state,
                states=demo["states"],
                camera_names=["agentview", "frontview", "robot0_eye_in_hand"],
                video_writer=video_writer,
                video_skip=1,
            )
            logging.info(
                f"Saved video of controller tracking eef poses to {save_video_path}"
            )

    if merge_demo_save_path is None:
        logging.info("No successful demos to merge.")
        return

    logging.info("Generating observations from merged demo states ...")
    render_camera_names = ["agentview", "robot0_eye_in_hand"]
    camera_height, camera_width = 84, 84
    depth = False
    # 构建状态转观测的参数对象
    dataset_states_to_obs_args = SimpleNamespace(
        dataset=merge_demo_save_path,
        output_name=f"{str(pathlib.Path(merge_demo_save_path).stem)}_obs.hdf5",
        n=None,  # 处理多少条数据？None 表示处理所有数据。
        shaped=False,
        camera_names=render_camera_names,
        camera_height=camera_height,
        camera_width=camera_width,
        depth=depth,
        done_mode=1,  # done = done or (t == traj_len)
        copy_rewards=False,  # 是否直接复制奖励和结束信号（这里设为 False，说明脚本会重新计算或者忽略）。
        copy_dones=False,
        exclude_next_obs=True,  # True:不保存 next_observation（下一帧观测）。
        compress=False,  # False:不压缩图片。
        use_actions=False,  # 是否重新计算动作（这里为 False，表示直接用原来的动作）。
    )

    if cfg.use_wandb:
        wandb.log(
            {
                "Final Success Rate": n_successes / trials,
                "Total successes": n_successes,
                "Total trials": trials,
            }
        )
        wandb.finish()  # 告诉 WandB 客户端：“本次实验结束了，关闭连接，停止同步。”

        # Append wandb run id to merge_demo_save_path's hdf5 file
        # 添加wandb运行链接
        if merge_demo_save_path is not None and cfg.wandb_run_url is not None:
            with h5py.File(merge_demo_save_path, "a") as file:
                file.attrs["wandb_run_url"] = cfg.wandb_run_url

    from robomimic.scripts.dataset_states_to_obs import dataset_states_to_obs

    # 执行图像渲染，并生成新的hdf5文件
    dataset_states_to_obs(dataset_states_to_obs_args)
    # Append wandb run id to merge_demo_save_path obs's hdf5 file
    # 添加wandb运行链接
    if merge_demo_save_path is not None and cfg.wandb_run_url is not None:
        with h5py.File(
            pathlib.Path(merge_demo_save_path).parent
            / f"{str(pathlib.Path(merge_demo_save_path).stem)}_obs.hdf5",
            "a",
        ) as file:
            file.attrs["wandb_run_url"] = cfg.wandb_run_url

    from scripts.dataset.mp4_from_h5 import Config as MP4H5Config
    from scripts.dataset.mp4_from_h5 import generate_videos_from_hdf5

    # 从带有图片的hdf5文件中为每一个demo生成视频
    generate_videos_from_hdf5(
        MP4H5Config(
            h5_file_path=pathlib.Path(merge_demo_save_path).parent
            / f"{str(pathlib.Path(merge_demo_save_path).stem)}_obs.hdf5",
            all_demos=True,
            fps=20,
        )
    )


if __name__ == "__main__":
    # load demonstrations file
    # tyro 比 argparse 还要更进一步，它不需要add_argument添加一个又一个参数，而是自动将命令行参数与config的属性相对应
    # 并且如果运行 `python your_script.py --help`会自动生成帮助文档
    tyro.cli(main)
