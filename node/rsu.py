import numpy as np
from typing import Set, Dict
from collections import deque


class Task:
    """简化任务类（与Vehicle/无人机中的Task兼容）"""

    def __init__(self, task_id: str, vehicle_id: int, compute_load: float):
        self.task_id = task_id
        self.vehicle_id = vehicle_id
        self.compute_load = compute_load
        self.status = "PENDING"  # 状态: PENDING / IN_PROGRESS / COMPLETED / FAILED
        self.finish_time = None  # 任务完成时间（可选）


class RSU:
    def __init__(self, rsu_id: int, position: np.ndarray, coverage_range: float, compute_capacity: float):
        """
        初始化RSU
        参数：
        rsu_id: RSU的唯一编号
        position: RSU的三维坐标 [x, y, z]
        coverage_range: RSU的覆盖范围（米）
        compute_capacity: RSU的总计算资源（GHz）
        """
        self.rsu_id = rsu_id
        self.position = np.array(position, dtype=np.float64)
        self.coverage_range = coverage_range
        self.compute_capacity = compute_capacity
        self.available_resources = compute_capacity  # 剩余计算资源
        self.km = 10e-27  # 有效电容系数
        # 新增状态跟踪
        self.covered_vehicles: Set[int] = set()  # 覆盖范围内的车辆ID集合

    def get_state(self) -> dict:
        """返回 RSU 的状态（用于SAC算法观察）"""
        return {
            "rsu_id": self.rsu_id,
            "position": tuple(self.position),
            "available_resources": self.available_resources,
            "covered_vehicles": list(self.covered_vehicles),
        }

    def update_covered_vehicles(self, vehicles: Dict[int, np.ndarray]):
        """更新覆盖范围内的车辆集合"""
        self.covered_vehicles.clear()
        for vid, pos in vehicles.items():
            distance = np.linalg.norm(self.position - pos)
            if distance <= self.coverage_range:
                self.covered_vehicles.add(vid)




    def reset(self):
        """重置 RSU 状态"""
        self.position = np.random.uniform(0, 1000, 3)
        self.available_resources = self.compute_capacity
        self.covered_vehicles.clear()

    def __repr__(self):
        return (f"RSU(ID={self.rsu_id}, Position={self.position}, "
                f"Coverage={self.coverage_range}m, "
                f"Available={self.available_resources:.2f}/{self.compute_capacity}GHz, ")