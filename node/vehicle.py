import numpy as np
import random
from enum import IntEnum

# 分类阈值配置
COMPUTE_THRESHOLD = 2000  # MB
LATENCY_THRESHOLD = 3  # s


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
        self.max_latency = random.uniform(1, 10)  # 秒
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
        self.direction = self._random_direction()

        self.compute_capacity = 8e8  # 8gHz
        self.tx_power = 0.1  # w
        self.km = 10e-27#车辆的有效电容系数
        self.task_queue = []  # 任务队列 [Task实例]

    def update_capacity(self):
        pass#待完善

    def _random_direction(self):
        dir = np.random.rand(2) - 0.5
        return dir / np.linalg.norm(dir)

    def move(self, time_step: float):
        if random.random() < 0.1:
            new_dir = self.direction + 0.2 * (np.random.rand(2) - 0.5)
            self.direction = new_dir / np.linalg.norm(new_dir)

        movement = self.direction * self.speed * time_step
        if movement.shape[0] == 2:
            # 在 z 方向补0
            movement = np.append(movement, 0)
        new_pos = self.position + movement
        new_pos = np.clip(new_pos, 0, 1000)

        if not np.array_equal(new_pos, self.position + movement):
            self.direction = -self.direction

        self.position = new_pos

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
