import numpy as np
import random
from enum import IntEnum

# 分类阈值配置
COMPUTE_THRESHOLD = 2000  # MB
LATENCY_THRESHOLD = 0.1  # s
# 10km × 10km 场景，五个十字路口（单位：米）
intersections = [
    np.array([5000, 5000, 0]),  # 主十字路口（中心）
    np.array([2500, 5000, 0]),  # 左侧十字路口
    np.array([7500, 5000, 0]),  # 右侧十字路口
    np.array([5000, 2500, 0]),  # 下侧十字路口
    np.array([5000, 7500, 0])  # 上侧十字路口
]

# 车道（水平和垂直）
lane_positions = []
for inter in intersections:
    lane_positions.append((inter - np.array([4000, 0, 0]), inter + np.array([4000, 0, 0])))  # 水平车道
    lane_positions.append((inter - np.array([0, 4000, 0]), inter + np.array([0, 4000, 0])))

class TaskType(IntEnum):
    RESOURCE_INTENSIVE = 1
    DELAY_SENSITIVE = 2
    PRIORITY = 3
    NORMAL = 4


class Task:
    def __init__(self, vehicle_id, task_id, timestamp):
        self.task_id = task_id
        self.vehicle_id = vehicle_id
        self.timestamp = timestamp
        self.data_size = random.uniform(100, 20000)  # MB
        self.max_latency = random.uniform(0.01, 1)  # 秒
        self.compute_load = 1000* self.data_size
        self.task_type = self._classify_task()
        self.offload_ration = 0.5#默认50%的任务会被卸载
        self.offload_targets = "local"  # 默认本地处理


    def _classify_task(self):
        if self.data_size >= COMPUTE_THRESHOLD and self.max_latency >= LATENCY_THRESHOLD:
            return TaskType.RESOURCE_INTENSIVE
        elif self.data_size < COMPUTE_THRESHOLD and self.max_latency < LATENCY_THRESHOLD:
            return TaskType.DELAY_SENSITIVE
        elif self.data_size >= COMPUTE_THRESHOLD and self.max_latency < LATENCY_THRESHOLD:
            return TaskType.PRIORITY
        else:
            return TaskType.NORMAL


class Vehicle:
    def __init__(self, vid: int):
        self.vid = vid
        self.position = np.array([np.random.uniform(0, 1000),
                                  np.random.uniform(0, 1000),
                                  0.0])  # [x, y, z]
        self.speed = np.random.uniform(15, 60)
        self.lane_positions = lane_positions  # 车道信息
        self.lane, self.direction = self._assign_lane_and_direction(lane_positions)

        self.compute_capacity = 8e8  # 8gHz
        self.tx_power = 0.1  # w
        self.km = 10e-27#车辆的有效电容系数
        self.task_queue = []  # 任务队列 [Task实例]

    def update_capacity(self):
        pass#待完善

    def _random_lane_position(self, lane_positions):
        """从车道中随机选择一个初始位置"""
        lane = random.choice(lane_positions)  # 选一个车道
        t = np.random.rand()
        return (1 - t) * lane[0] + t * lane[1]  # 在车道范围内随机分布

    def _assign_lane_and_direction(self, lane_positions):
        """选择最近的车道，并确定行驶方向"""
        closest_lane = None
        min_dist = float('inf')
        for lane in lane_positions:
            mid_point = (lane[0] + lane[1]) / 2
            dist = np.linalg.norm(self.position - mid_point)
            if dist < min_dist:
                min_dist = dist
                closest_lane = lane
        # 计算行驶方向
        direction = closest_lane[1] - closest_lane[0]
        direction = direction[:2] / np.linalg.norm(direction[:2])  # 归一化
        return closest_lane, direction
    def move(self, time_step: float):
        """沿车道移动"""
        movement = np.array([self.direction[0], self.direction[1], 0]) * self.speed * time_step
        new_pos = self.position + movement

        # 判断是否超出当前车道范围
        lane_min = np.min(np.vstack([self.lane[0], self.lane[1]]), axis=0)
        lane_max = np.max(np.vstack([self.lane[0], self.lane[1]]), axis=0)
        if np.any(new_pos < lane_min) or np.any(new_pos > lane_max):
            self.respawn()
        else:
            self.position = new_pos

    def respawn(self):
        """车辆到达边界后，在车道上重新生成位置"""
        self.position = self._random_lane_position(self.lane_positions)
        self.lane, self.direction = self._assign_lane_and_direction(self.lane_positions)

    def generate_task(self, timestamp: float):
        task = Task(vehicle_id=self.vid, task_id=f"V{self.vid}-T{timestamp}", timestamp=timestamp)
        self.task_queue.append(task)
        return task


    def current_task(self):
        return self.task_queue[0] if self.task_queue else None

    def get_state(self):
        return {
            "vid": self.vid,
            "position": tuple(self.position),
            "speed": self.speed,
            "self": self.compute_capacity,  # GHz # GHz
            "task_queue": [task.task_id for task in self.task_queue],
        }

    def __repr__(self):
        return f"Vehicle(vid={self.vid}, tasks={len(self.task_queue)}, pos={self.position})"
