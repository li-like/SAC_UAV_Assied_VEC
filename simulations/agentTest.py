import numpy as np
import torch
import math

from torch.distributions import Normal


# 假设的 Network 类
class Network:
    def __init__(self):
        self.vehicles = [Vehicle(i) for i in range(2)]  # 2 辆车
        self.rsus = [RSU(i) for i in range(3)]  # 3 个 RSU
        self.lower_uavs = [LowerUAV(i) for i in range(2)]  # 2 个 LUAV
        self.upper_uavs = [UpperUAV(i) for i in range(1)]  # 1 个 HUAV
        self.base_stations = [BaseStation(i) for i in range(1)]  # 1 个 BS

# 假设的节点类
class Vehicle:
    def __init__(self, vid):
        self.vid = vid
        self.position = np.array([100 * vid, 200, 0])
        self.compute_capacity = 100
        self.tx_power = 10
        self.current_task = lambda: Task(compute_load=1000, data_size=1e6)

class RSU:
    def __init__(self, rsu_id):
        self.rsu_id = rsu_id
        self.position = np.array([50 * rsu_id, 50, 0])
        self.compute_capacity = 200
        self.tx_power = 20

class LowerUAV:
    def __init__(self, uav_id):
        self.uav_id = uav_id
        self.position = np.array([150 * uav_id, 150, 50])
        self.compute_capacity = 150
        self.tx_power = 15

class UpperUAV:
    def __init__(self, uav_id):
        self.uav_id = uav_id
        self.position = np.array([200, 200, 100])
        self.compute_capacity = 300
        self.tx_power = 25

class BaseStation:
    def __init__(self, bs_id):
        self.bs_id = bs_id
        self.position = np.array([300, 300, 0])
        self.compute_capacity = 500
        self.tx_power = 30

# 假设的 Task 类
class Task:
    def __init__(self, compute_load, data_size):
        self.compute_load = compute_load
        self.data_size = data_size

# 假设的 ChannelModel 类
class ChannelModel:
    def vehicle_to_Clostrsu_time(self, vehicle_pos, rsu_pos, task, compute_capacity, tx_power):
        return 5.2  # 固定返回值，用于测试

    def vehicle_to_luav_time(self, vehicle_pos, luav_pos, task, compute_capacity, tx_power):
        return 3.8  # 固定返回值，用于测试

    def luav_rsu_time(self, vehicle_pos, huav_pos, rsu_pos, task, compute_capacity, tx_power_veh, tx_power_huav):
        return 4.5  # 固定返回值，用于测试

    def luav_bs_time(self, vehicle_pos, huav_pos, bs_pos, task, compute_capacity, tx_power_veh, tx_power_huav):
        return 6.0  # 固定返回值，用于测试

# 假设的 SACAgent 类
class SACAgent:
    def __init__(self):
        self.network = Network()
        self.actor = lambda x: (torch.zeros(30), torch.zeros(30))  # 固定返回值，用于测试

    def select_action(self, state_vector):
        if np.isnan(state_vector).any() or np.isinf(state_vector).any():
            raise ValueError("State vector contains NaN or inf!")
        state_tensor = torch.FloatTensor(state_vector).unsqueeze(0)
        mu, log_std = self.actor(state_tensor)
        std = log_std.exp()
        dist = Normal(mu, std)
        raw_action = dist.rsample()  # 原始采样值
        raw_action = torch.tanh(raw_action)  # 限制在 [-1,1]
        raw_action = raw_action.detach().numpy()[0]
        # 定义卸载目标和计算资源分配的标签
        offload_targets = ["local", "Closet_RSU", "LUAV", "HUAV_RSU", "HUAV_BS"]
        offload_ratio_targets = ["local", "Closet_RSU", "LUAV", "HUAV_RSU", "HUAV_BS"]
        compute_targets = ["local", "Closet_RSU", "LUAV", "HUAV_RSU", "HUAV_BS"]
        # 将 raw_action 解析为每辆车的策略
        strategies = np.reshape(raw_action, (len(self.network.vehicles), 15))
        # 创建一个总字典来存储所有车的策略
        task_strategies = {}
        for i in range(len(self.network.vehicles)):
            # 每辆车的策略
            vehicle_strategy = {
                "offload_targets": dict(zip(offload_targets, strategies[i, :5])),  # 卸载目标
                "offload_ratios": dict(zip(offload_ratio_targets, strategies[i, 5:10])),  # 卸载比例
                "compute_allocations": dict(zip(compute_targets, strategies[i, 10:]))  # 计算资源分配
            }
            # 将每辆车的策略添加到总字典中
            task_strategies[f"vehicle_{i}"] = vehicle_strategy
            # 映射 offload_targets
            task_strategies[f"vehicle_{i}"]["mapped_targets"] = self.map_offload_targets(
                vehicle_strategy["offload_ratios"],
                vehicle_strategy["offload_targets"],
                self.find_closest_rsu(self.network.vehicles[i].position),  # 每辆车最近的 RSU 的 ID
            )
        # 打印结果
        print("Task Strategies:", task_strategies)
        return task_strategies

    def compute_task_delay(self, task_strategies):
        """
        计算每个任务的总时延，并选择最大时延作为任务的最终时延
        Args:
            task_strategies (dict): 卸载策略字典
        Returns:
            dict: 每个任务的最大时延，例如 {"vehicle_1": 5.2, "vehicle_2": 3.8, ...}
        """
        task_delays = {}
        channel = ChannelModel()  # 初始化信道模型

        for vehicle_id, strategy in task_strategies.items():
            # 获取车辆对象
            vehicle = self.network.vehicles[int(vehicle_id.split("_")[1]) - 1]

            # 初始化每种卸载方式的时延
            delays = []

            # 遍历卸载目标
            for target, ratio in strategy["offload_ratios"].items():
                if ratio > 0:
                    # 获取节点信息
                    node_id = strategy["mapped_targets"][target]
                    node = self._get_node(target, node_id, vehicle)
                    print("target", target)
                    print("ratio", ratio)
                    if node:
                        # 计算时延
                        if target == "local":
                            delay = vehicle.current_task().compute_load / vehicle.compute_capacity
                        elif target == "Closet_RSU":
                            delay = channel.vehicle_to_Clostrsu_time(
                                vehicle.position, node.position, vehicle.current_task(),
                                node.compute_capacity, vehicle.tx_power
                            )
                        elif target == "LUAV":
                            delay = channel.vehicle_to_luav_time(
                                vehicle.position, node.position, vehicle.current_task(),
                                node.compute_capacity, vehicle.tx_power
                            )
                        elif target == "HUAV_RSU":
                            delay = channel.luav_rsu_time(
                                vehicle.position, node.position, node.position, vehicle.current_task(),
                                node.compute_capacity, vehicle.tx_power, node.tx_power
                            )
                        elif target == "HUAV_BS":
                            delay = channel.luav_bs_time(
                                vehicle.position, node.position, node.position, vehicle.current_task(),
                                node.compute_capacity, vehicle.tx_power, node.tx_power
                            )
                        else:
                            raise ValueError(f"Invalid offload target: {target}")

                        # 记录时延
                        delays.append(delay)

            # 选择最大时延作为任务的最终时延
            if delays:
                task_delays[vehicle_id] = max(delays)
            else:
                task_delays[vehicle_id] = 0  # 如果没有卸载目标，时延为 0

        print("Task Delays:", task_delays)
        return task_delays

    def _get_node(self, target_type, node_id, vehicle):
        """根据类型和ID获取节点对象"""
        if target_type == "Closet_RSU":
            return self.network.rsus[self.find_closest_rsu(vehicle.position)] if node_id < len(self.network.rsus) else None
        elif target_type == "LUAV":
            return self.network.lower_uavs[node_id] if node_id < len(self.network.lower_uavs) else None
        elif target_type == "HUAV_RSU":
            return self.network.rsus[node_id] if node_id < len(self.network.rsus) else None
        elif target_type == "HUAV_BS":
            return self.network.upper_uavs[0] if node_id < len(self.network.base_stations) else None
        elif target_type == "local":
            return vehicle if node_id < len(self.network.vehicles) else None
        return None

    def find_closest_rsu(self, vehicle_position):
        """
        找到距离车辆最近的 RSU 节点
        Args:
            vehicle_position (np.ndarray): 车辆的位置，格式为 [x, y, z]
        Returns:
            int: 最近的 RSU 节点 ID
            float: 最近的距离
        """
        if not self.network.rsus:
            raise ValueError("没有可用的 RSU 节点")
        # 计算车辆与所有 RSU 的距离
        distances = [
            np.linalg.norm(rsu.position - vehicle_position)
            for rsu in self.network.rsus
        ]
        # 找到最小距离的 RSU 节点 ID
        closest_rsu_id = np.argmin(distances)
        closest_distance = distances[closest_rsu_id]
        return closest_rsu_id

    def map_offload_targets(self, offload_ratios, offload_targets, closest_rsu_id):
        mapped_targets = {}
        # 映射 local
        if offload_ratios["local"] > 0:
            mapped_targets["local"] = 1  # local 固定映射为 1

        # 映射 Closet_RSU
        if offload_ratios["Closet_RSU"] > 0:
            mapped_targets["Closet_RSU"] = closest_rsu_id  # 映射为最近的 RSU

        # 映射 LUAV
        if offload_ratios["LUAV"] > 0:
            luav_id = math.ceil(len(self.network.lower_uavs) * offload_targets["LUAV"])  # 根据比例映射
            mapped_targets["LUAV"] = luav_id

        # 映射 HUAV_RSU
        if offload_ratios["HUAV_RSU"] > 0:
            rsu_id = math.ceil(len(self.network.rsus) * offload_targets["HUAV_RSU"])  # 根据比例映射
            mapped_targets["HUAV_RSU"] = rsu_id

        # 映射 HUAV_BS
        if offload_ratios["HUAV_BS"] > 0:
            mapped_targets["HUAV_BS"] = 0  # 固定映射为第一个 BS

        return mapped_targets

# 测试代码
if __name__ == "__main__":
    # 初始化 SACAgent
    agent = SACAgent()

    # 生成测试用的 raw_action
    raw_action = np.random.rand(30)  # 假设 raw_action 是一个长度为 30 的数组

    # 调用 select_action
    task_strategies = agent.select_action(raw_action)

    # 调用 compute_task_delay
    task_delays = agent.compute_task_delay(task_strategies)