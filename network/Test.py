import matplotlib.pyplot as plt
import numpy as np
import torch
import time

# 导入你定义的 Network 和 SACAgent 类
from environment import Network
from SAC_agent import SACAgent

def main():
    # ---------------------------
    # 1. 环境与代理初始化
    # ---------------------------
    # 使用你定义的 Network 类作为环境
    network = Network()
    # 重置环境，生成初始状态
    # 注意：网络代码中 state_dim 与动作空间由 _calculate_real_state_dim 和 _calculate_action_dim 得到
    # 创建 SACAgent，并传入环境实例
    agent = SACAgent(network)

    # ---------------------------
    # 2. 训练超参数设置
    # ---------------------------
    num_episodes = 100          # 总 episode 数
    max_timesteps = 100        # 每个 episode 内的最大步数
    episode_rewards = []       # 存储每个 episode 的总奖励
    episode_lengths = []       # 存储每个 episode 的步数

    # ---------------------------
    # 3. 开始训练循环
    # ---------------------------
    for episode in range(num_episodes):
        # 重置环境，获取初始状态
        timesteps = 0
        state = network.reset_nodes(30,4,1,1)
        done = False
        episode_reward = 0
        # 用于记录本 episode 内车辆轨迹（这里只以第一辆车为例）
        vehicle_trajectory = []

        while not done and timesteps < max_timesteps:
            # 获取当前全局状态（状态为 numpy 数组）
            state = network.get_global_state()
            # print(f"Step {timesteps}, State: {state}")  # 打印当前状态

            # 选择动作，返回值为各车辆的卸载策略字典
            task_strategies,luav_strategies = agent.select_action(state)
            # print(f"Step {timesteps}, Action: {action}")  # 打印选择的动作

            # 执行动作，环境更新：agent.step(action) 内部调用 _apply_action（目前为 pass）和 _update_environment
            next_state, reward, done, info = agent.step(task_strategies,luav_strategies)
            # print(f"Step {timesteps}, Next State: {next_state}, Reward: {reward}, Done: {done}")  # 打印下一状态、奖励和 done 标志
            combined_strategies = {**task_strategies, **luav_strategies}
            # 存储经验并进行更新（你的 SACAgent 中已有 replay_buffer 和 update 方法）
            agent.store_experience(state, combined_strategies, reward, next_state, done)
            agent.update(batch_size=128)  # 根据实际情况选择 batch_size

            episode_reward += reward
            timesteps += 1

            # 记录本 timestep 所有车辆的位置信息
            # 这里以第一辆车位置为例（车辆的 position 是一个 numpy 数组）
            vehicle_trajectory.append(network.vehicles[0].position.copy())
            if done:
                print(f"Episode {episode_reward} finished. Total reward: {episode_reward}")

        episode_rewards.append(episode_reward)
        episode_lengths.append(timesteps)
        print(f"Episode {episode+1}/{num_episodes}, Reward: {episode_reward}, Timesteps: {timesteps}")

        # ---------------------------
        # 4. 可视化本 episode 车辆运动轨迹（以第一辆车为例）
        # ---------------------------
        vehicle_trajectory = np.array(vehicle_trajectory)  # shape: (timesteps, 3)
        plt.figure(figsize=(8, 6))
        plt.plot(vehicle_trajectory[:, 0], vehicle_trajectory[:, 1], marker='o', label='Trajectory')
        plt.scatter(vehicle_trajectory[0, 0], vehicle_trajectory[0, 1], color='green', label='Start')
        plt.scatter(vehicle_trajectory[-1, 0], vehicle_trajectory[-1, 1], color='red', label='End')
        plt.xlabel("X Position")
        plt.ylabel("Y Position")
        plt.title(f"Vehicle 0 Trajectory (Episode {episode+1})")
        plt.legend()
        plt.close()

    # ---------------------------
    # 5. 可视化训练进度
    # ---------------------------
    def smooth_curve(data, alpha=0.3):
        smoothed = []
        last = data[0]  # 初始值
        for point in data:
            last = alpha * point + (1 - alpha) * last  # 计算指数移动平均
            smoothed.append(last)
        return smoothed

    # 计算平滑后的曲线
    smoothed_rewards = smooth_curve(episode_rewards, alpha=0.3)
    smoothed_lengths = smooth_curve(episode_lengths, alpha=0.3)

    plt.figure(figsize=(10, 5))
    plt.plot(range(1, num_episodes + 1), episode_rewards, marker='o', label="Original", alpha=0.3)
    plt.plot(range(1, num_episodes + 1), smoothed_rewards, marker='o', label="Smoothed", color='blue')
    plt.xlabel("Episode")
    plt.ylabel("Total Reward")
    plt.title("Training Progress: Episode Rewards")
    plt.legend()
    plt.grid(True)
    plt.savefig("training_progress.png")
    plt.close()

    plt.figure(figsize=(10, 5))
    plt.plot(range(1, num_episodes + 1), episode_lengths, marker='o', color='orange', alpha=0.3, label="Original")
    plt.plot(range(1, num_episodes + 1), smoothed_lengths, marker='o', color='red', label="Smoothed")
    plt.xlabel("Episode")
    plt.ylabel("Episode Length (timesteps)")
    plt.title("Training Progress: Episode Lengths")
    plt.legend()
    plt.grid(True)
    plt.savefig("episode_lengths.png")
    plt.close()

    # ---------------------------
    # 6. 最后可视化整个网络环境（调用你在 Network 类中定义的 visualize 方法）
    # ---------------------------
    network.visualize()

    print("Training complete. Plots saved and final environment visualized.")

def print_task_details(task):
    details = (
        "\n-------------------------\n"
        "      Task Details       \n"
        "-------------------------\n"
        f"Task ID         : {task.task_id}\n"
        f"Vehicle ID      : {task.vehicle_id}\n"
        f"Timestamp       : {task.timestamp:.2f}\n"
        f"Data Size       : {task.data_size:.2f} MB\n"
        f"Max Latency     : {task.max_latency:.2f} s\n"
        f"Compute Load    : {task.compute_load:.2f}\n"
        f"Task Type       : {task.task_type.name}\n"
        f"Offload Targets : {task.offload_targets}\n"
        "-------------------------\n"
    )
    print(details)

if __name__ == "__main__":
    main()
