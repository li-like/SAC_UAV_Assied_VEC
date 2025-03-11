import numpy as np
from typing import Set, Dict, Optional
from collections import deque


class Task:
    """简化任务类（与Vehicle/无人机中的Task兼容）"""

    def __init__(self, task_id: str, vehicle_id: int, compute_load: float):
        self.task_id = task_id
        self.vehicle_id = vehicle_id
        self.compute_load = compute_load
        self.status = "PENDING"  # 状态: PENDING / IN_PROGRESS / COMPLETED / FAILED
        self.finish_time = None  # 任务完成时间（可选）


class BaseStation:
    def __init__(self, bs_id: int, position: np.ndarray, coverage_range: float, compute_capacity: float):
        """
        初始化基站
        参数：
        bs_id: 基站的唯一编号
        position: 基站的三维坐标 [x, y, z]
        coverage_range: 基站的覆盖范围（米）
        compute_capacity: 基站的总计算资源（GHz）
        """
        self.bs_id = bs_id
        self.position = np.array(position, dtype=np.float64)
        self.coverage_range = coverage_range
        self.compute_capacity = compute_capacity
        self.available_resources = compute_capacity  # 剩余计算资源
    def get_state(self) -> dict:
        """返回基站的状态（用于SAC算法观察）"""
        return {
            "bs_id": self.bs_id,
            "position": tuple(self.position),
            "available_resources": self.available_resources,
        }

    def reset(self):
        """重置基站状态"""
        self.position = np.random.uniform(0, 1000, 3)
        self.available_resources = self.compute_capacity

    def __repr__(self):
        return (f"BaseStation(ID={self.bs_id}, Position={self.position}, "
                f"Coverage={self.coverage_range}m, "
                f"Available={self.available_resources:.2f}/{self.compute_capacity}GHz, ")