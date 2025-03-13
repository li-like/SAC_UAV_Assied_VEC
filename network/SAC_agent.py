import math
from typing import Dict, Optional
from scipy.special import softmax
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions import Normal
import numpy as np
from network.channel import ChannelModel


class SACAgent:
    def __init__(self, network, hidden_dim=256, lr=1e-4, gamma=0.99, tau=0.01, alpha=0.01, weight_decay=1e-5):
        self.network = network  # 直接使用外部的网络环境实例
        state_dim = network.state_dim
        action_dim = network.action_dim
        self.actor = Actor(state_dim, action_dim, hidden_dim)
        self.critic1 = Critic(state_dim, action_dim, hidden_dim)
        self.critic2 = Critic(state_dim, action_dim, hidden_dim)
        self.target_critic1 = Critic(state_dim, action_dim, hidden_dim)
        self.target_critic2 = Critic(state_dim, action_dim, hidden_dim)
        self.target_critic1.load_state_dict(self.critic1.state_dict())
        self.target_critic2.load_state_dict(self.critic2.state_dict())

        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=lr, weight_decay=weight_decay)
        self.critic1_optimizer = optim.Adam(self.critic1.parameters(), lr=lr, weight_decay=weight_decay)
        self.critic2_optimizer = optim.Adam(self.critic2.parameters(), lr=lr, weight_decay=weight_decay)

        self.gamma = gamma
        self.tau = tau
        self.alpha = alpha
        self.replay_buffer = ReplayBuffer(max_size=100000)

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
        task_action = raw_action[:15 *len(self.network.vehicles)]  # 前 15 * num_vehicles 长度的部分
        luav_action = raw_action[15 * len(self.network.vehicles):]  # 后 5 长度的部分
        offload_targets = ["local", "Closet_RSU", "LUAV", "HUAV_RSU", "HUAV_BS"]
        offload_ratio_targets = ["local", "Closet_RSU", "LUAV", "HUAV_RSU", "HUAV_BS"]
        compute_targets = ["local", "Closet_RSU", "LUAV", "HUAV_RSU", "HUAV_BS"]
        luav_move_strategies = np.reshape(luav_action, (len(self.network.lower_uavs), 2))
        luav_strategies = {}
        for i in range(len(self.network.lower_uavs)):
            luav_strategy = {
                "direction": np.clip(luav_move_strategies[i, 0], 0, 1),  # 限制方向在 [0, 1]
                "speed": np.clip(luav_move_strategies[i, 1], 0, 50)  # 限制速度在 [0, 50]
            }
            luav_strategies[f"luav_{i}"] = luav_strategy
        # 将 task_action 解析为每辆车的策略
        strategies = np.reshape(task_action, (len(self.network.vehicles), 15))
        # 创建一个总字典来存储所有车的策略
        task_strategies = {}
        node_resource_usage = {}
        for i in range(len(self.network.vehicles)):
            # 每辆车的策略
            # 对 offload_ratios 和 compute_allocations 使用 Sigmoid 限制到 (0,1)
            offload_ratios_raw = strategies[i, 5:10]
            compute_allocations_raw = strategies[i, 10:]

            offload_ratios = 1 / (1 + np.exp(-offload_ratios_raw))
            compute_allocations = 1 / (1 + np.exp(-compute_allocations_raw))
            offload_ratios = softmax(offload_ratios, axis=-1)  # 使用scipy的softmax
            vehicle_strategy = {
                "offload_targets": dict(zip(offload_targets, strategies[i, :5])),  # 卸载目标
                "offload_ratios": dict(zip(offload_ratio_targets, offload_ratios)),  # 卸载比例
                "compute_allocations": dict(zip(compute_targets, compute_allocations))  # 计算资源分配
            }
            # 映射 offload_targets
            mapped_targets = self.map_offload_targets(vehicle_strategy["offload_targets"],self.find_closest_rsu(self.network.vehicles[i].position))
            vehicle_strategy["offload_targets"] = mapped_targets
            for target, ratio in vehicle_strategy["offload_ratios"].items():
                if ratio > 0 and target != "local":
                    node_id = vehicle_strategy["offload_targets"][target]
                    if "RSU" in target:
                        node_key = f"RSU_{node_id}"  # 节点唯一标识符，例如 "RSU_1"
                    else:
                        node_key = f"{target}_{node_id}"  # 节点唯一标识符，例如 "RSU_1"
                    compute_allocation = vehicle_strategy["compute_allocations"][target]
                    # 更新节点资源使用总量
                    if node_key in node_resource_usage:
                        node_resource_usage[node_key] += compute_allocation
                    else:
                        node_resource_usage[node_key] = compute_allocation
            task_strategies[f"vehicle_{i}"] = vehicle_strategy
            # 检查并修正资源分配
            # 检查并修正资源分配
        # 修正资源分配（遍历所有车辆策略）
        for node_key, total_allocation in node_resource_usage.items():
            if total_allocation > 1:
                scale_factor = 1 / total_allocation
                for vehicle_strategy in task_strategies.values():
                    for target in vehicle_strategy["offload_ratios"]:
                        if target == "local":
                            continue
                        current_node_id = vehicle_strategy["offload_targets"][target]
                        if "RSU" in target:
                            current_node_key = f"RSU_{current_node_id}"  # 节点唯一标识符，例如 "RSU_1"
                        else:
                            current_node_key = f"{target}_{current_node_id}"  # 节点唯一标识符，例如 "RSU_1"
                        if current_node_key == node_key:
                            vehicle_strategy["compute_allocations"][target] *= scale_factor
                node_resource_usage[node_key] = 1  # 确保总量为1
            # 打印结果
        return task_strategies,luav_strategies

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
        total_energy = 0
        for vehicle_id, strategy in task_strategies.items():
            # 仅处理键名以 "vehicle_" 开头的条目
            if not vehicle_id.startswith("vehicle_"):
                continue
            try:
                idx = int(vehicle_id.split("_")[1])
            except ValueError:
                print(f"Warning: vehicle_id {vehicle_id} 格式错误，跳过该项。")
                continue
            # 使用索引直接获取车辆
            vehicle = self.network.vehicles[idx]
            delays = []
            # 遍历卸载目标
            for target, ratio in strategy["offload_ratios"].items():
                vehicle.current_task().offload_ratio = ratio
                vehicle.current_task().offload_target = target
                if ratio > 0:
                    # 获取节点信息
                    node_id = strategy["offload_targets"][target]
                    node = self._get_node(target, node_id,vehicle)
                    if node!=None:
                        # 计算时延
                        if target == "local":
                            delay = vehicle.current_task().compute_load / vehicle.compute_capacity*strategy["compute_allocations"]["local"]
                            total_energy += delay* pow(vehicle.compute_capacity*strategy["compute_allocations"]["local"],3)*vehicle.km
                        elif target == "Closet_RSU":
                            transmit_delay,compute_delay = channel.vehicle_to_Clostrsu_time(
                                vehicle.position, node.position, vehicle.current_task(),
                                node.compute_capacity*strategy["compute_allocations"]["Closet_RSU"], vehicle.tx_power
                            )
                            delay = transmit_delay + compute_delay
                            total_energy += (compute_delay * pow(
                                node.compute_capacity*strategy["compute_allocations"]["Closet_RSU"], 3) * node.km+vehicle.tx_power*transmit_delay)
                        elif target == "LUAV":
                            transmit_delay,compute_delay = channel.vehicle_to_luav_time(
                                vehicle.position, node.position, vehicle.current_task(),
                                node.compute_capacity*strategy["compute_allocations"]["LUAV"], vehicle.tx_power
                            )
                            delay = transmit_delay + compute_delay
                            total_energy += (compute_delay * pow(
                                node.compute_capacity * strategy["compute_allocations"]["LUAV"],
                                3) * node.km + vehicle.tx_power * transmit_delay)
                        elif target == "HUAV_RSU":
                            delay = channel.huav_rsu_time(
                                vehicle.position, node.position, node.position, vehicle.current_task(),
                                node.compute_capacity*strategy["compute_allocations"]["HUAV_RSU"], vehicle.tx_power
                            )
                            total_energy += (delay * pow(
                                node.compute_capacity * strategy["compute_allocations"]["HUAV_RSU"],
                                3) * node.km)
                        elif target == "HUAV_BS":
                            delay = channel.huav_bs_time(
                                vehicle.position, node.position, node.position, vehicle.current_task(),
                                node.compute_capacity*strategy["compute_allocations"]["HUAV_BS"], vehicle.tx_power
                            )
                            total_energy += (delay * pow(
                                node.compute_capacity * strategy["compute_allocations"]["HUAV_BS"],
                                3) * node.km)
                        else:
                            raise ValueError(f"Invalid offload target: {target}")

                        # 记录时延
                        delays.append(delay)

            # 选择最大时延作为任务的最终时延
            if delays:
                task_delays[vehicle_id] = max(delays)
            else:
                task_delays[vehicle_id] = 0  # 如果没有卸载目标，时延为 0

        return task_delays,total_energy
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

    def map_offload_targets(self, offload_targets,closest_rsu_id):
        """
        将 offload_targets 产生的数值映射为对应节点的 id。
        这里假设：
          - "local" 表示车辆本地，固定返回 0（或车辆自身 id，根据需求调整）。
          - "Closet_RSU" 映射到 RSU 列表索引（0 到 len(rsus)-1）。
          - "LUAV" 映射到 lower_uavs 列表索引（0 到 len(lower_uavs)-1）。
          - "HUAV_RSU" 映射到 RSU 列表索引（0 到 len(rsus)-1）。
          - "HUAV_BS" 映射到 base_stations 列表索引（0 到 len(base_stations)-1）。
        """
        mapped = {}

        # local 固定映射为 0，表示本地计算
        mapped["local"] = 0

        # 映射 Closet_RSU
        rsu_count = len(self.network.rsus)
        if rsu_count > 0:
            # 将 offload_targets["Closet_RSU"] 从 [-1, 1] 归一化到 [0, 1]
            norm = (offload_targets["Closet_RSU"] + 1) / 2.0
            mapped["Closet_RSU"] = closest_rsu_id
        else:
            mapped["Closet_RSU"] = None

        # 映射 LUAV
        luav_count = len(self.network.lower_uavs)
        if luav_count > 0:
            norm = (offload_targets["LUAV"] + 1) / 2.0
            mapped["LUAV"] = min(int(norm * luav_count), luav_count - 1)
        else:
            mapped["LUAV"] = None

        # 映射 HUAV_RSU
        # 假设此处仍使用 RSU 节点
        if rsu_count > 0:
            norm = (offload_targets["HUAV_RSU"] + 1) / 2.0
            mapped["HUAV_RSU"] = min(int(norm * rsu_count), rsu_count - 1)
        else:
            mapped["HUAV_RSU"] = None

        # 映射 HUAV_BS
        bs_count = len(self.network.base_stations)
        if bs_count > 0:
            norm = (offload_targets["HUAV_BS"] + 1) / 2.0
            mapped["HUAV_BS"] = min(int(norm * bs_count), bs_count - 1)
        else:
            mapped["HUAV_BS"] = None

        return mapped


    def get_reward(self,task_strategies,luav_strategies):
        total_reward = 0
        total_task_delay = 0  # 初始化总任务时延
        total_energy = self._apply_action(task_strategies,luav_strategies)
        task_delay, other_energy = self.compute_task_delay(task_strategies)
        total_task_delay += sum(task_delay.values())  # 累加所有车辆的任务时延
        total_energy += other_energy

        for vehicle_id, strategy in task_strategies.items():
            offload_ratios = strategy["offload_ratios"]
            if not vehicle_id.startswith("vehicle_"):
                continue
            try:
                idx = int(vehicle_id.split("_")[1])
            except ValueError:
                print(f"Warning: vehicle_id {vehicle_id} 格式错误，跳过该项。")
                continue
            # 使用索引直接获取车辆
            vehicle = self.network.vehicles[idx]
            task = vehicle.current_task()
            task_attribute = task.task_type
            max_latency = task.max_latency  # 任务最大容忍时延
            if_Classify = False #检查任务是否分类正确

            current_delay = task_delay.get(vehicle_id, float('inf'))  # 若无数据则默认超时
            # 3. 根据任务属性计算奖励
            if task_attribute == "RESOURCE_INTENSIVE":
                # HUAV_RSU + HUAV_BS 的卸载比例是否大于 0.7
                huav_rsu_bs_ratio = offload_ratios.get("HUAV_RSU", 0) + offload_ratios.get("HUAV_BS", 0)
                if huav_rsu_bs_ratio > 0.7:
                    if_Classify = True
                else:
                    if_Classify = False
            elif task_attribute == "DELAY_SENSITIVE":
                # local + Closet_RSU + LUAV 的卸载比例是否大于 0.6
                local_rsu_luav_ratio = offload_ratios.get("local", 0) + offload_ratios.get("Closet_RSU",                                                                              0) + offload_ratios.get(
                    "LUAV", 0)
                if local_rsu_luav_ratio > 0.6:
                    if_Classify = True
                else:
                    if_Classify = False
            elif task_attribute == "PRIORITY":
                # Closet_RSU + LUAV 的卸载比例是否大于 0.7
                rsu_luav_ratio = offload_ratios.get("Closet_RSU", 0) + offload_ratios.get("LUAV", 0)
                if rsu_luav_ratio > 0.7:
                    if_Classify = True
                else:
                    if_Classify = False
            else:
                if_Classify = True
            # 1. 检查卸载比例总和是否为 1
            sum_ratios = sum(offload_ratios.values())
            #奖励计算部分
            ratio_error = abs(sum_ratios - 1.0)
            # 奖励与误差成反比，误差越小奖励越高
            total_reward -= ratio_error * 1  # 调整系数可根据实际情况修改
            # 假设 current_delay 表示任务时延，max_latency 为最大容忍时延
            if current_delay <= max_latency:
                delay_reward = (max_latency - current_delay) / max_latency * 1  # 奖励值在 0～100 之间
                total_reward += delay_reward
            else:
                total_reward -= 2  # 超时惩罚
            #检查是否某一项卸载比例为 1
            if any(math.isclose(ratio, 1.0, rel_tol=1e-5) for ratio in offload_ratios.values()):
                total_reward -= 1  # 鼓励部分卸载，给予负奖励
            else:
                total_reward += 1
            if not if_Classify:
                total_reward -= 1
            else:
                total_reward += 1
        total_reward -= (total_energy*0.5 +total_task_delay*20)
        return total_reward

    def step(self, task_strategies,luav_strategies):
        # 1. 执行动作,在rewrad中执行
        # 2. 计算当前奖励
        reward = self.get_reward(task_strategies,luav_strategies)
        # 3. 更新环境状态（例如车辆移动、任务生成）
        self._update_environment(self.network.time_step)
        # 4. 获取下一状态
        next_state = self.network.get_global_state()
        # 5. 检查是否终止（例如时间步长超过阈值）
        done = self.network.time_step >= 1000  # 假设最大时间步为 1000
        self.network.time_step += 1
        # 6. 返回结果
        print(f"reward:{reward}")
        return next_state, reward, done, {}
    def get_energy(self,task_strategies):
        total_energy = 0

    def _apply_action(self, task_strategies,luav_strategies):

        # 遍历每个底层无人机（LUAV）
        luav_cosumu_energy = 0
        for luav_key, luav_action in luav_strategies.items():
            luav_id = int(luav_key.split("_")[1])  # 提取编号
            direction = luav_action["direction"]
            speed = luav_action["speed"]
            # 调用 luav_move 方法更新位置
            luav_cosumu_energy += (self.network.lower_uavs[luav_id].
                                  consume_energy(self.network.lower_uavs[luav_id].luav_move(1, direction, speed),1))
        return luav_cosumu_energy+self.network.upper_uavs[0].consume_energy()


    def _update_environment(self,timestep):
        """
        更新网络环境状态（车辆移动、任务清除和重新生成）
        """
        # 1. 车辆沿车道随机移动
        for vehicle in self.network.vehicles:
            vehicle.move(1.0)  # 假设时间步长为 1.0

        # 2. 清除所有未完成的任务
        for vehicle in self.network.vehicles:
            vehicle.task_queue.clear()

        # 3. 重新生成新任务
        for vehicle in self.network.vehicles:
            # 示例：生成一个计算负载为 1000、数据大小为 1e6 的任务
            vehicle.generate_task(timestamp=float(timestep))

    def soft_update(self):
        # soft update的机制
        for target_param, param in zip(self.target_critic1.parameters(), self.critic1.parameters()):
            target_param.data.copy_(self.tau * param.data + (1.0 - self.tau) * target_param.data)
        for target_param, param in zip(self.target_critic2.parameters(), self.critic2.parameters()):
            target_param.data.copy_(self.tau * param.data + (1.0 - self.tau) * target_param.data)

    def update(self, batch_size=256):
        if len(self.replay_buffer) < batch_size:
            return

        states, actions, rewards, next_states, dones = self.replay_buffer.sample(batch_size)
        states = torch.FloatTensor(states)
        actions = torch.FloatTensor(actions)
        rewards = torch.FloatTensor(rewards).unsqueeze(1)
        next_states = torch.FloatTensor(next_states)
        dones = torch.FloatTensor(dones).unsqueeze(1)

        with torch.no_grad():
            next_actions, next_log_probs = self.actor.sample(next_states)
            target_q1 = self.target_critic1(next_states, next_actions)
            target_q2 = self.target_critic2(next_states, next_actions)
            target_q = torch.min(target_q1, target_q2) - self.alpha * next_log_probs
            target_q = rewards + (1 - dones) * self.gamma * target_q

        current_q1 = self.critic1(states, actions)
        current_q2 = self.critic2(states, actions)
        critic1_loss = F.mse_loss(current_q1, target_q)
        critic2_loss = F.mse_loss(current_q2, target_q)

        self.critic1_optimizer.zero_grad()
        critic1_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.critic1.parameters(), max_norm=0.5)
        self.critic1_optimizer.step()

        self.critic2_optimizer.zero_grad()
        critic2_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.critic2.parameters(), max_norm=0.5)
        self.critic2_optimizer.step()

        new_actions, log_probs = self.actor.sample(states)
        q1 = self.critic1(states, new_actions)
        q2 = self.critic2(states, new_actions)
        actor_loss = (self.alpha * log_probs - torch.min(q1, q2)).mean()

        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.actor.parameters(), max_norm=0.5)
        self.actor_optimizer.step()
        # # 打印 actor 网络部分参数的范数
        # for name, param in self.actor.named_parameters():
        #     print(f"Actor {name} norm: {param.data.norm().item()}")
        # # 同理打印 critic 参数
        # for name, param in self.critic1.named_parameters():
        #     print(f"Critic1 {name} norm: {param.data.norm().item()}")
        # 软更新目标网络
        self.soft_update()

    def flatten_action(self, action_dict):
        """
        将 task_strategies 和 luav_strategies 合并后的动作字典扁平化为 numpy 数组。
        """
        flat = []

        # 先处理车辆（vehicle_x）
        vehicle_keys = sorted([k for k in action_dict.keys() if k.startswith("vehicle_")],
                              key=lambda x: int(x.split("_")[1]))
        for vehicle_key in vehicle_keys:
            vehicle_action = action_dict[vehicle_key]
            for key in ["offload_targets", "offload_ratios", "compute_allocations"]:
                subdict = vehicle_action[key]
                # 确保所有 key 按固定顺序展开
                for subkey in ["local", "Closet_RSU", "LUAV", "HUAV_RSU", "HUAV_BS"]:
                    flat.append(float(subdict[subkey]))

        # 再处理低空无人机（luav_x）
        luav_keys = sorted([k for k in action_dict.keys() if k.startswith("luav_")], key=lambda x: int(x.split("_")[1]))
        for luav_key in luav_keys:
            luav_action = action_dict[luav_key]
            # 按顺序添加 direction 和 speed
            flat.append(float(luav_action["direction"]))
            flat.append(float(luav_action["speed"]))

        return np.array(flat, dtype=np.float32)

    def store_experience(self, state, action, reward, next_state, done):
        # 如果动作是字典，则先扁平化
        if isinstance(action, dict):
            action = self.flatten_action(action)
        self.replay_buffer.push(state, action, reward, next_state, done)


class Actor(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=256):
        super(Actor, self).__init__()
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        nn.init.xavier_uniform_(self.fc1.weight)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        nn.init.xavier_uniform_(self.fc2.weight)
        self.mu = nn.Linear(hidden_dim, action_dim)
        nn.init.xavier_uniform_(self.mu.weight)
        self.log_std = nn.Linear(hidden_dim, action_dim)
        nn.init.xavier_uniform_(self.log_std.weight)

    def forward(self, state):
        if torch.isnan(state).any():
            print("Warning: input state contains NaN!")
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        mu = self.mu(x)

        log_std = self.log_std(x)
        log_std = torch.clamp(log_std, -20, 2)
        return mu, log_std

    def sample(self, state):
        mu, log_std = self.forward(state)
        std = log_std.exp()
        dist = Normal(mu, std)
        action = dist.rsample()
        log_prob = dist.log_prob(action).sum(-1, keepdim=True)
        action = torch.tanh(action)
        return action, log_prob


class Critic(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=256):
        super(Critic, self).__init__()
        self.fc1 = nn.Linear(state_dim + action_dim, hidden_dim)
        nn.init.xavier_uniform_(self.fc1.weight)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        nn.init.xavier_uniform_(self.fc2.weight)
        self.fc3 = nn.Linear(hidden_dim, 1)
        nn.init.xavier_uniform_(self.fc3.weight)

    def forward(self, state, action):
        x = torch.cat([state, action], dim=1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)


class ReplayBuffer:
    def __init__(self, max_size=100000):
        self.buffer = []
        self.max_size = max_size

    def push(self, state, action, reward, next_state, done):
        if len(self.buffer) >= self.max_size:
            self.buffer.pop(0)
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        indices = np.random.randint(0, len(self.buffer), batch_size)
        states, actions, rewards, next_states, dones = zip(*[self.buffer[i] for i in indices])
        return np.array(states), np.array(actions), np.array(rewards), np.array(next_states), np.array(dones)

    def __len__(self):
        return len(self.buffer)
