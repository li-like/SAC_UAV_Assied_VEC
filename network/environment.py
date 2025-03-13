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
            num_vehicles=30,
            num_lower_uavs=4,
            num_upper_uavs=1,
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
        action_dim = 15*len(self.vehicles)+2*len(self.lower_uavs)#(每个任务都有五种选择，每个选择有3个动作,卸载节点，卸载比例，计算资源分配)
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

        # 低空无人机状态：位置 (x, y, z), 速度, 资源, 电量比例
        for lower_uav in self.lower_uavs:
            pos = lower_uav.position.tolist()
            lower_uav_state = pos + [
                lower_uav.compute_capacity,
                lower_uav.energy
            ]
            state.extend(lower_uav_state)

        # 高空无人机状态：位置 (x, y, z), 速度, 资源比例, 电量比例
        for upper_uav in self.upper_uavs:
            pos = upper_uav.position.tolist()
            upper_uav_state = pos + [
                upper_uav.compute_capacity,
                upper_uav.energy
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
        # 将状态列表转换为 numpy 数组，确保所有元素为 float32
        return np.array(state, dtype=np.float32)

    def _calculate_real_state_dim(self):
        """通过实际生成状态向量计算维度"""

        # 车辆状态维度
        vehicle_state_dim = (3 + 1 + 1 + 3 * 1)  # 位置 + 速度 + 资源 + 任务队列（每个任务3个维度）
        # 低空无人机状态维度
        lower_uav_state_dim = 3  + 1 + 1  # 位置  + 资源 + 电量
        # 高空无人机状态维度
        upper_uav_state_dim = 3   + 1+1  # 位置 +  资源比例+ 电量
        # RSU状态维度
        rsu_state_dim = 3 + 1 + 1 + 1  # 位置 + 资源 + 覆盖范围 + 覆盖车辆数
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


    def reset_nodes(self, num_vehicles, num_lower_uavs, num_upper_uavs,num_base_stations=1):
        """初始化所有节点，并按照新的10km*10km布局生成合理的位置"""
        # 清空已有节点
        self.vehicles = []
        self.lower_uavs = []
        self.upper_uavs = []
        self.rsus = []
        self.base_stations = []

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
            lane_positions.append((inter - np.array([0, 4000, 0]), inter + np.array([0, 4000, 0])))  # 垂直车道

        # 初始化车辆（分布在十字车道上）
        for i in range(num_vehicles):
            lane = random.choice(lane_positions)
            position = lane[0] + np.random.rand() * (lane[1] - lane[0])
            vehicle = Vehicle(vid=i)
            vehicle.position = position
            vehicle.position[2] = 0  # 固定高度
            vehicle.task_queue.clear()
            vehicle.generate_task(vehicle.vid)
            self.add_vehicle(vehicle)

        # 初始化低空无人机（在十字路口附近）
        for i in range(num_lower_uavs):
            intersection = random.choice(intersections)
            position = intersection + np.random.uniform(-500, 500, 3)
            position[2] = 50  # 固定高
            lower_uav = LowerUAV(uav_id=i, position=position)
            lower_uav.task_queue.clear()
            lower_uav.energy = 500000
            self.add_lower_uav(lower_uav)

        # 初始化高空无人机（在主十字路口上方）
        for i in range(num_upper_uavs):
            intersection = intersections[0]
            position = intersection + np.random.uniform(-50, 50, 3)
            position[2] = 200  # 固定高度
            upper_uav = UpperUAV(uav_id=i, position=position)
            upper_uav.task_queue.clear()
            upper_uav.energy = 500000
            self.add_upper_uav(upper_uav)

        # 初始化 RSU（均匀分布在路边，共8个）
        rsu_positions = []
        # **1. 在十字路口布置 RSU**
        rsu_positions.extend(intersections)
        interval = 2000  # RSU 间隔 2000m
        coverage_range = 800  # RSU 覆盖范围 800m
        # **2. 计算所有车道的 RSU 部署点**
        roads = [
            {"fixed_axis": "y", "values": [5000, 2500, 7500], "varying_axis": "x", "range": (1000, 9000)},  # 水平方向
            {"fixed_axis": "x", "values": [5000, 2500, 7500], "varying_axis": "y", "range": (1000, 9000)}  # 垂直方向
        ]

        for road in roads:
            for fixed_value in road["values"]:
                for coord in range(road["range"][0], road["range"][1] + 1, interval):
                    if road["fixed_axis"] == "y":
                        pos = np.array([coord, fixed_value, 0])
                    else:
                        pos = np.array([fixed_value, coord, 0])

                    # 避免 RSU 和十字路口重复
                    if not any(np.array_equal(pos, inter) for inter in intersections):
                        rsu_positions.append(pos)

        # **3. 初始化 RSU**
        for i, position in enumerate(rsu_positions):
            rsu = RSU(rsu_id=i, position=position, coverage_range=coverage_range, compute_capacity=50e8)
            self.add_rsu(rsu)

        # 初始化基站（放置在边界，靠近四个角落）
        base_station_positions = [
            np.array([1000, 1000, 0]),  # 左下角
            np.array([9000, 1000, 0]),  # 右下角
            np.array([1000, 9000, 0]),  # 左上角
            np.array([9000, 9000, 0])  # 右上角
        ]
        for i in range(num_base_stations):
            position = random.choice(base_station_positions)
            base_station = BaseStation(bs_id=i, position=position, coverage_range=2000, compute_capacity=70e8)
            self.add_base_station(base_station)

        self.time_step = 0
        return self.get_global_state()


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
        """优化可视化的网络环境"""
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='3d')

        # 设置视角
        ax.view_init(elev=30, azim=45)

        # 场景范围
        scene_size = 10000  # 10km * 10km
        ax.set_xlim([0, scene_size])
        ax.set_ylim([0, scene_size])
        ax.set_zlim([0, 3000])  # 最高 3km 适用于无人机

        # 定义十字路口坐标
        intersections = [
            np.array([5000, 5000, 0]),  # 主十字路口
            np.array([2500, 5000, 0]),  # 左侧十字路口
            np.array([7500, 5000, 0]),  # 右侧十字路口
            np.array([5000, 2500, 0]),  # 下侧十字路口
            np.array([5000, 7500, 0])  # 上侧十字路口
        ]

        # 车道（水平和垂直）
        lane_positions = [
            ((0, 5000, 0), (10000, 5000, 0)),  # 水平主干道
            ((0, 2500, 0), (10000, 2500, 0)),  # 水平次干道
            ((0, 7500, 0), (10000, 7500, 0)),  # 水平次干道
            ((5000, 0, 0), (5000, 10000, 0)),  # 垂直主干道
            ((2500, 0, 0), (2500, 10000, 0)),  # 垂直次干道
            ((7500, 0, 0), (7500, 10000, 0))  # 垂直次干道
        ]

        # 绘制车道
        for lane in lane_positions:
            x_vals = [lane[0][0], lane[1][0]]
            y_vals = [lane[0][1], lane[1][1]]
            z_vals = [lane[0][2], lane[1][2]]
            ax.plot(x_vals, y_vals, z_vals, 'k-', linewidth=2, alpha=0.7)

        # 绘制十字路口
        ax.scatter(*zip(*intersections), c='black', marker='X', s=100, label="Intersection")

        # 统一存储绘制的数据，避免重复 label
        legend_added = set()

        # 绘制车辆
        for vehicle in self.vehicles:
            label = "Vehicle" if "Vehicle" not in legend_added else None
            ax.scatter(vehicle.position[0], vehicle.position[1], vehicle.position[2],
                       c='r', marker='o', s=30, alpha=0.8, label=label)
            legend_added.add("Vehicle")

        # 绘制低空无人机
        for lower_uav in self.lower_uavs:
            label = "Lower UAV" if "Lower UAV" not in legend_added else None
            ax.scatter(lower_uav.position[0], lower_uav.position[1], lower_uav.position[2],
                       c='b', marker='^', s=80, alpha=0.8, label=label)
            legend_added.add("Lower UAV")

        # 绘制高空无人机
        for upper_uav in self.upper_uavs:
            label = "Upper UAV" if "Upper UAV" not in legend_added else None
            ax.scatter(upper_uav.position[0], upper_uav.position[1], upper_uav.position[2],
                       c='g', marker='v', s=100, alpha=0.8, label=label)
            legend_added.add("Upper UAV")

        # 绘制 RSU
        for rsu in self.rsus:
            label = "RSU" if "RSU" not in legend_added else None
            ax.scatter(rsu.position[0], rsu.position[1], rsu.position[2],
                       c='y', marker='s', s=120, alpha=0.9, label=label)
            legend_added.add("RSU")

        # 绘制基站
        for base_station in self.base_stations:
            label = "Base Station" if "Base Station" not in legend_added else None
            ax.scatter(base_station.position[0], base_station.position[1], base_station.position[2],
                       c='m', marker='D', s=150, alpha=0.9, label=label)
            legend_added.add("Base Station")

        # 设置坐标轴标签
        ax.set_xlabel('X Coordinate (m)')
        ax.set_ylabel('Y Coordinate (m)')
        ax.set_zlabel('Z Coordinate (m)')

        # 添加网格线
        ax.grid(True, linestyle='--', alpha=0.5)

        # 统一图例
        ax.legend()

        # 显示图形
        plt.show()


# 创建网络实例并可视化
network = Network()
network.visualize()


