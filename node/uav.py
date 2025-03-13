import math

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






class LowerUAV(UAV):
    def __init__(self, uav_id: int, position: np.ndarray):
        super().__init__(
            uav_id=uav_id,
            position=position,
            compute_capacity=15e8,  #15GHZ
            tx_power=0.2  #w
        )
        self.direction = np.array([0.0, 0.0], dtype=np.float64)
        self.communication_range = 500.0  # 通信半径 (米)
        self.max_vehicles = 10  # 最大关联车辆数
        self.km = 10e-27  # 有效电容系数
        self.m_uav = 9.65  # uav质量/kg
        self.p0=158.76 #无人机悬停状态时叶片的轮廓功率w
        self.p1 = 88.63  # 无人机悬停状态时感应功率w

    def luav_move(self, time_step: float,direction,speed):
        """移动逻辑"""
        theta = direction * np.pi * 2  # 角度
        dis_fly = speed * time_step  # 飞行距离
        dx_uav = dis_fly * math.cos(theta)
        dy_uav = dis_fly * math.sin(theta)
        # 更新位置
        self.position[0] += dx_uav
        self.position[1] += dy_uav
        self.position[2] = 50
        return dis_fly
    def consume_energy(self, dis_fly: float, duration: float):
        """消耗能量"""
        if dis_fly == 0:#悬停状态
            e_consume = (self.p0 + self.p1) * duration
        else:
            e_consume = (dis_fly / duration) ** 2 * self.m_uav * duration * 0.5
        self.energy = max(0.0, self.energy-e_consume)#更新能量
        return e_consume


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
        self.p0 = 79.8  # 无人机悬停状态时叶片的轮廓功率w
        self.p1 = 60.2  # 无人机悬停状态时感应功率w
        self.km = 10e-27  # 有效电容系数
    def consume_energy(self):
        e_consume = (self.p0 + self.p1) * 1
        self.energy = max(0.0, self.energy - e_consume)  # 更新能量
        return e_consume


