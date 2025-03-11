import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
from node.bs import BaseStation
from node.rsu import RSU
from node.uav import LowerUAV, UpperUAV
from node.vehicle import Vehicle
import random



class Network:
    def __init__(self):
        # 初始化节点列表
        self.vehicles = []
        self.lower_uavs = []
        self.upper_uavs = []
        self.rsus = []
        self.base_stations = []

        # 初始化时间步长
        self.time_step = 0

        # 先生成节点，再计算状态和动作维度
        self.reset_nodes(
            num_vehicles=15,
            num_lower_uavs=1,
            num_upper_uavs=1,
            num_rsus=5,
            num_base_stations=1
        )
        # 计算真实状态维度
        self.state_dim = self._calculate_real_state_dim()
        # 动作空间维度
        self.action_dim = self._calculate_action_dim()

    def _calculate_action_dim(self):
        """
        计算动作空间的维度
        """
        action_dim = 15*len(self.vehicles)#(每个任务都有五种选择，每个选择有3个动作,卸载节点，卸载比例，计算资源分配)
        return action_dim

    def get_global_state(self) -> np.ndarray:
        """返回适配SAC算法的全局状态向量"""
        state = []
        max_task_queue_length = 1  # 固定任务队列长度
        # 车辆状态：位置 (x, y, z), 速度, 资源比例, 任务队列（每个任务3个维度：compute_load, max_latency, data_size）
        for vehicle in self.vehicles:
            pos = vehicle.position.tolist()
            base_info = [vehicle.speed, vehicle.compute_capacity]
            # 获取任务队列数据
            task_loads = [task.compute_load for task in vehicle.task_queue]
            task_maxDelay = [task.max_latency for task in vehicle.task_queue]
            task_datasize = [task.data_size for task in vehicle.task_queue]

            vehicle_state = pos + base_info + task_loads + task_maxDelay + task_datasize
            state.extend(vehicle_state)

        # 低空无人机状态：位置 (x, y, z), 速度, 资源比例, 电量比例
        for lower_uav in self.lower_uavs:
            pos = lower_uav.position.tolist()
            lower_uav_state = pos + [
                lower_uav.speed,
                lower_uav.compute_capacity,
                lower_uav.energy / 100.0
            ]
            state.extend(lower_uav_state)

        # 高空无人机状态：位置 (x, y, z), 速度, 资源比例, 电量比例
        for upper_uav in self.upper_uavs:
            pos = upper_uav.position.tolist()
            upper_uav_state = pos + [
                upper_uav.compute_capacity,
                upper_uav.energy / 100.0
            ]
            state.extend(upper_uav_state)

        # RSU状态：位置 (x, y, z), 资源比例, 覆盖范围, 覆盖车辆数量
        for rsu in self.rsus:
            pos = rsu.position.tolist()
            covered_count = len(rsu.covered_vehicles)
            rsu_state = pos + [
                rsu.compute_capacity,
                rsu.coverage_range,
                covered_count
            ]
            state.extend(rsu_state)

        # 基站状态：位置 (x, y, z), 资源比例, 覆盖范围
        for base_station in self.base_stations:
            pos = base_station.position.tolist()
            bs_state = pos + [
                base_station.compute_capacity,
                base_station.coverage_range
            ]
            state.extend(bs_state)

        # 添加时间步长
        state.append(self.time_step)
        print(f"state:{state}")
        # 将状态列表转换为 numpy 数组，确保所有元素为 float32
        return np.array(state, dtype=np.float32)

    def _calculate_real_state_dim(self):
        """通过实际生成状态向量计算维度"""

        # 车辆状态维度
        vehicle_state_dim = (3 + 1 + 1 + 3 * 1)  # 位置 + 速度 + 资源比例 + 任务队列（每个任务3个维度）
        # 低空无人机状态维度
        lower_uav_state_dim = 3 + 1 + 1 + 1  # 位置 + 速度 + 资源比例 + 电量比例
        # 高空无人机状态维度
        upper_uav_state_dim = 3  + 1 + 1  # 位置 + 速度 + 资源比例 + 电量比例
        # RSU状态维度
        rsu_state_dim = 3 + 1 + 1 + 1  # 位置 + 资源比例 + 覆盖范围 + 覆盖车辆数
        # 基站状态维度
        base_station_state_dim = 3 + 1 + 1  # 位置 + 资源比例 + 覆盖范围
        # 计算总维度
        total_state_dim = (
                len(self.vehicles) * vehicle_state_dim +
                len(self.lower_uavs) * lower_uav_state_dim +
                len(self.upper_uavs) * upper_uav_state_dim +
                len(self.rsus) * rsu_state_dim +
                len(self.base_stations) * base_station_state_dim +
                1  # 时间步长
        )

        return total_state_dim

    def add_vehicle(self, vehicle: Vehicle):
        self.vehicles.append(vehicle)

    def add_lower_uav(self, uav: LowerUAV):
        self.lower_uavs.append(uav)

    def add_upper_uav(self, uav: UpperUAV):
        self.upper_uavs.append(uav)

    def add_rsu(self, rsu: RSU):
        self.rsus.append(rsu)

    def add_base_station(self, base_station: BaseStation):
        self.base_stations.append(base_station)

    def reset_nodes(self, num_vehicles, num_lower_uavs, num_upper_uavs, num_rsus,
                         num_base_stations):
        """初始化所有节点，并按照布局生成合理的位置"""
        # 清空已有节点列表，防止重复添加
        self.vehicles = []
        self.lower_uavs = []
        self.upper_uavs = []
        self.rsus = []
        self.base_stations = []

        lane_positions = [
            (np.array([100, 500, 0]), np.array([900, 500, 0])),  # 水平车道
            (np.array([500, 100, 0]), np.array([500, 900, 0]))  # 垂直车道
        ]
        intersections = [np.array([500, 500, 0])]  # 十字路口位置

        # 初始化车辆（在车道上）
        for i in range(num_vehicles):
            lane = random.choice(lane_positions)
            position = lane[0] + np.random.rand() * (lane[1] - lane[0])
            vehicle = Vehicle(vid=i)
            vehicle.position = position
            vehicle.position[2] = 0  # 固定高度
            vehicle.direction=vehicle._random_direction()
            vehicle.task_queue.clear()
            vehicle.generate_task(vehicle.vid)
            self.add_vehicle(vehicle)

        # 初始化低空无人机（尽量在十字路口附近）
        for i in range(num_lower_uavs):
            position = intersections[0] + np.random.uniform(-300, 300, 3)
            position[2] = 50  # 固定高度
            lower_uav = LowerUAV(uav_id=i, position=position, speed=10)
            lower_uav.task_queue.clear()
            lower_uav.energy=100
            self.add_lower_uav(lower_uav)

        # 初始化高空无人机（在十字路口上方）
        for i in range(num_upper_uavs):
            position = intersections[0] + np.random.uniform(-10, 10, 3)
            position[2] = 200  # 固定高度
            upper_uav = UpperUAV(uav_id=i, position=position)
            upper_uav.task_queue.clear()
            upper_uav.energy = 100
            self.add_upper_uav(upper_uav)

        # 初始化 RSU（在路边）
        for i in range(num_rsus):
            lane = random.choice(lane_positions)
            position = lane[0] + np.random.rand() * (lane[1] - lane[0])
            position[2] = 0  # 固定高度
            position += np.random.uniform(-100, 100, 3) * np.array([1, 1, 0])
            rsu = RSU(rsu_id=i, position=position, coverage_range=300, compute_capacity=50e8)
            self.add_rsu(rsu)

        # 初始化基站（在边缘区域）
        for i in range(num_base_stations):
            position = np.array([random.choice([100, 900]), random.choice([100, 900]), 0])
            position[2] = 0  # 固定高度
            base_station = BaseStation(bs_id=i, position=position, coverage_range=500, compute_capacity=70e8)
            self.add_base_station(base_station)
        self.time_step = 0
        # self.print_for_init()
        return self.get_global_state()
    # def reset(self):
    #     """重置环境并返回初始状态"""
    #     for vehicle in self.vehicles:
    #         vehicle.reset()
    #     for lower_uav in self.lower_uavs:
    #         lower_uav.reset()
    #     for upper_uav in self.upper_uavs:
    #         upper_uav.reset()
    #     for rsu in self.rsus:
    #         rsu.reset()
    #     for base_station in self.base_stations:
    #         base_station.reset()
    #     self.time_step = 0


    def print_for_init(self):
        print("初始化节点信息完成：")
        print(
            f"共{len(self.vehicles)}辆车辆、{len(self.lower_uavs)}架低空无人机、{len(self.upper_uavs)}架高空无人机、{len(self.rsus)}个RSU、{len(self.base_stations)}个基站。\n")

        print("=== 车辆列表 ===")
        for vehicle in self.vehicles:
            pos = vehicle.position
            print(f"Vehicle {vehicle.vid} "
                  f"位置: [{pos[0]:.2f}, {pos[1]:.2f}, {pos[2]:.2f}]"
                  f" 速度: {vehicle.speed} m/s"
                  f" 任务列表: {vehicle.task_queue} ")

        print("\n=== 低空无人机 ===")
        for uav in self.lower_uavs:
            pos = uav.position
            print(f"Lower UAV {uav.uav_id} "
                  f"位置: [{pos[0]:.2f}, {pos[1]:.2f}, {pos[2]:.2f}] "
                  f"速度: {uav.speed} m/s")

        print("\n=== 高空无人机 ===")
        for uav in self.upper_uavs:
            pos = uav.position
            print(f"Upper UAV {uav.uav_id} "
                  f"位置: [{pos[0]:.2f}, {pos[1]:.2f}, {pos[2]:.2f}]")

        print("\n=== RSU 列表 ===")
        for rsu in self.rsus:
            pos = rsu.position
            print(
                f"RSU {rsu.rsu_id} 位置: [{pos[0]:.2f}, {pos[1]:.2f}, {pos[2]:.2f}] "
                f"覆盖范围: {rsu.coverage_range} m "
                f"计算能力: {rsu.compute_capacity:.0e} ops")

        print("\n=== 基站列表 ===")
        for bs in self.base_stations:
            pos = bs.position
            print(f"BaseStation {bs.bs_id} 位置: [{pos[0]:.2f}, {pos[1]:.2f}, "
                  f"{pos[2]:.2f}] 覆盖范围: {bs.coverage_range} m 计算能力: {bs.compute_capacity:.0e} ops")
    def visualize(self):
        """可视化初始化后的网络环境"""
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')

        # 绘制车辆节点
        for vehicle in self.vehicles:
            ax.scatter(vehicle.position[0], vehicle.position[1], vehicle.position[2], c='r', marker='o',
                       label="Vehicle")

        # 绘制低空无人机节点
        for lower_uav in self.lower_uavs:
            ax.scatter(lower_uav.position[0], lower_uav.position[1], lower_uav.position[2], c='b', marker='^',
                       label="Lower UAV")

        # 绘制高空无人机节点
        for upper_uav in self.upper_uavs:
            ax.scatter(upper_uav.position[0], upper_uav.position[1], upper_uav.position[2], c='g', marker='v',
                       label="Upper UAV")

        # 绘制RSU节点
        for rsu in self.rsus:
            ax.scatter(rsu.position[0], rsu.position[1], rsu.position[2], c='y', marker='s', label="RSU")

        # 绘制基站节点
        for base_station in self.base_stations:
            ax.scatter(base_station.position[0], base_station.position[1], base_station.position[2], c='m', marker='D',
                       label="Base Station")

        # 设置坐标轴标签
        ax.set_xlabel('X Coordinate')
        ax.set_ylabel('Y Coordinate')
        ax.set_zlabel('Z Coordinate')

        # 防止标签重复
        handles, labels = ax.get_legend_handles_labels()
        by_label = dict(zip(labels, handles))
        ax.legend(by_label.values(), by_label.keys())

        # 显示图形
        plt.show()


# 创建网络实例并可视化
network = Network()
network.visualize()


