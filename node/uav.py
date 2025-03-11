import numpy as np
import random
from collections import deque
from enum import Enum



class UAV:
    def __init__(self, uav_id: int, position: np.ndarray, compute_capacity: float, tx_power: float):
        self.uav_id = uav_id
        self.position = position.astype(np.float64)  # 强制浮点类型
        self.compute_capacity = compute_capacity  # 总计算资源 (GHz)
        self.available_resources = compute_capacity  # 剩余计算资源
        self.tx_power = tx_power  # 发射功率 w
        self.task_queue = deque()  # 待处理任务队列 (先进先出)
        # uav电池电量: 500kJ. ref: Mobile Edge Computing via a UAV-Mounted Cloudlet: Optimization of Bit Allocation and Path Planning
        self.energy = 500000

    def get_state(self) -> dict:
        """返回无人机状态字典"""
        return {
            "uav_id": self.uav_id,
            "position": tuple(self.position),
            "energy": self.energy,
            "task_count": len(self.task_queue),
            "available_resources": self.available_resources
        }

    def reset(self):
        """重置无人机状态"""
        self.position = np.random.uniform(0, 1000, 3)
        self.available_resources = self.compute_capacity
        self.task_queue.clear()
        self.energy = 500000

    def consume_energy(self, energy_cost: float, duration: float):
        """消耗能量"""
        self.energy -= energy_cost * duration
        self.energy = max(0.0, self.energy)




class LowerUAV(UAV):
    def __init__(self, uav_id: int, position: np.ndarray, speed: float):
        super().__init__(
            uav_id=uav_id,
            position=position,
            compute_capacity=15e8,  #15GHZ
            tx_power=0.2  #w
        )
        self.speed = speed
        self.direction = np.array([0.0, 0.0], dtype=np.float64)
        self.communication_range = 500.0  # 通信半径 (米)
        self.max_vehicles = 10  # 最大关联车辆数
        self.km = 10e-27  # 有效电容系数

    def move(self, time_step: float):
        """移动逻辑"""
        if random.random() < 0.1:
            # 10%概率随机调整方向
            delta = np.random.uniform(-0.2, 0.2, size=2)
            self.direction += delta
            self.direction /= np.linalg.norm(self.direction)

        displacement = self.direction * self.speed * time_step
        self.position += displacement
        self.position = np.clip(self.position, 0.0, 1000.0)

        # 能量消耗
        self.consume_energy(0.1, time_step)  # 移动能耗


class UpperUAV(UAV):
    def __init__(self, uav_id: int, position: np.ndarray):
        super().__init__(
            uav_id=uav_id,
            position=position,
            compute_capacity=3e9,  #30ghz
            tx_power=0.2  # w
        )
        self.env_info = {}  # 环境信息缓存
        self.relay_threshold = 50  # 中继信号阈值 (dBm)




