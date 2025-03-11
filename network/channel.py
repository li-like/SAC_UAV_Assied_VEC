import numpy as np


class ChannelModel:
    """
    信道模型类，实现无人机辅助车联网中的各种信道计算
    包含以下主要功能：
    1. 车到RSU的非视距(NLoS)信道计算
    2. 车到LUAV的视距(LoS)信道计算
    3. 中继信道计算（车->HUAV->RSU/BS）
    """

    def __init__(self, cfg=None):
        """
        初始化默认信道参数
        """
        self.beta0 = 1e-5  # 参考距离信道增益 (1m处)
        self.N0 = 1e-13  # 噪声功率谱密度 (W/Hz)
        self.B_rsu = 10e6  # 车-RSU带宽 (Hz)
        self.B_uav = 20e6  # 车-UAV带宽 (Hz)
        self.alpha_los = 2.0  # LoS路径损耗指数
        self.alpha_nlos = 3.5  # NLoS路径损耗指数
        self.shadow_mu = 0  # 阴影衰落对数均值
        self.shadow_sigma = 4  # 阴影衰落对数标准差
        self.return_delta = 0.1  # 返回比例
        self.HUAV_TX_POWER = 0.2  # HUAV发射功率

        if cfg:  # 允许通过配置文件覆盖参数
            self.__dict__.update(cfg)

    def vehicle_to_Clostrsu_time(self, vehicle_pos, Clost_rsu_pos, task, f_rsu_0, P_tx):
        """
        车到RSU的非视距信道计算
        """
        distance = np.linalg.norm(np.array(vehicle_pos) - np.array(Clost_rsu_pos))
        h = (self.beta0 ) / (distance)
        if distance==0:
            raise ValueError("Invalid distance!")

        rate = self.B_rsu * np.log2(1 + h*P_tx/self.N0)
        transmit_delay = task.offload_ratio * task.data_size / rate
        compute_delay = task.offload_ratio * task.compute_load / f_rsu_0
        if rate==0:
            raise ValueError("Invalid rate!")
        return transmit_delay + compute_delay

    def vehicle_to_luav_time(self, vehicle_pos, luav_pos, task, f_luav, P_tx):
        """
        车到LUAV的视距信道计算
        """
        d = np.linalg.norm(np.array(vehicle_pos) - np.array(luav_pos))
        h = self.beta0 / (d )
        SNR = (P_tx * self.beta0 ) / (self.N0 )
        rate = self.B_uav * np.log2(1 + SNR)

        transmit_delay = task.offload_ratio * task.data_size / rate
        compute_delay = task.offload_ratio * task.compute_load / f_luav
        if rate==0:
            raise ValueError("Invalid rate!")
        return transmit_delay + compute_delay

    def luav_rsu_time(self, vehicle_pos, huav_pos, rsu_pos, task, f_rsu, P_tx_veh):
        """
        中继信道计算（车->HUAV->RSU/BS）: 车到高空无人机，再到RSU
        """
        d_vh = np.linalg.norm(np.array(vehicle_pos) - np.array(huav_pos))
        h_vh = self.beta0 / (d_vh)
        d_hr = np.linalg.norm(np.array(huav_pos) - np.array(rsu_pos))
        h_hr = self.beta0 / (d_hr)

        SNR_vh = (P_tx_veh * self.beta0) / (self.N0)
        rate_vh = self.B_uav * np.log2(1 + SNR_vh)
        transmit_delay_vh = task.offload_ratio * task.data_size / rate_vh

        SNR_hr = (self.HUAV_TX_POWER * self.beta0) / (self.N0)
        rate_hr = self.B_uav * np.log2(1 + SNR_hr)
        transmit_delay_hr = task.offload_ratio * task.data_size / rate_hr

        compute_delay = task.offload_ratio * task.compute_load / f_rsu
        return_delay = self.return_delta * (transmit_delay_vh + transmit_delay_hr)
        if rate_hr==0 or rate_vh==0:
            raise ValueError("Invalid rate!")
        return transmit_delay_vh + transmit_delay_hr + compute_delay + return_delay

    def luav_bs_time(self, vehicle_pos, huav_pos, bs_pos, task, f_bs, P_tx_veh):
        """
        中继信道计算（车->HUAV->BS）
        """
        d_vh = np.linalg.norm(np.array(vehicle_pos) - np.array(huav_pos))
        h_vh = self.beta0 / (d_vh)
        d_hr = np.linalg.norm(np.array(huav_pos) - np.array(bs_pos))
        h_hr = self.beta0 / (d_hr)

        SNR_vh = (P_tx_veh * self.beta0) / (self.N0)
        rate_vh = self.B_uav * np.log2(1 + SNR_vh)
        transmit_delay_vh = task.offload_ratio * task.data_size / rate_vh

        SNR_hr = (self.HUAV_TX_POWER * self.beta0) / (self.N0 )
        rate_hr = self.B_uav * np.log2(1 + SNR_hr)
        transmit_delay_hb = task.offload_ratio * task.data_size / rate_hr
        if rate_hr==0 or rate_vh==0:
            raise ValueError("Invalid rate!")
        compute_delay = task.offload_ratio * task.compute_load / f_bs
        return_delay = self.return_delta * (transmit_delay_vh + transmit_delay_hb)
        return transmit_delay_vh + transmit_delay_hb + compute_delay + return_delay

    def calculate_total_delay(self, vehicle_pos, task, offload_target, offload_ratio, network):
        """
        计算任务的总时延
        """
        # 将卸载比例赋值给任务对象属性
        task.offload_ratio = offload_ratio
        if offload_target == "Local":

            return task.compute_load*task.offload_ratio / network.vehicles[task.vehicle_id].compute_capacity

        elif offload_target == "NearestRSU":
            nearest_rsu = min(network.rsus, key=lambda rsu: np.linalg.norm(rsu.position - vehicle_pos))
            return self.vehicle_to_Clostrsu_time(vehicle_pos, nearest_rsu.position, task,
                                            nearest_rsu.compute_capacity, network.vehicles[task.vehicle_id].tx_power)

        elif offload_target == "LowerUAV":
            nearest_lower_uav = min(network.lower_uavs, key=lambda uav: np.linalg.norm(uav.position - vehicle_pos))
            return self.vehicle_to_luav_time(vehicle_pos, nearest_lower_uav.position, task,
                                             nearest_lower_uav.compute_capacity, network.vehicles[task.vehicle_id].tx_power)

        elif offload_target == "UpperUAV_RSU":
            nearest_upper_uav = min(network.upper_uavs, key=lambda uav: np.linalg.norm(uav.position - vehicle_pos))
            nearest_rsu = min(network.rsus, key=lambda rsu: np.linalg.norm(rsu.position - vehicle_pos))
            return self.luav_rsu_time(vehicle_pos, nearest_upper_uav.position, nearest_rsu.position, task,
                                      nearest_rsu.compute_capacity, network.vehicles[task.vehicle_id].tx_power,
                                      nearest_upper_uav.tx_power)

        elif offload_target == "UpperUAV_BS":
            nearest_upper_uav = min(network.upper_uavs, key=lambda uav: np.linalg.norm(uav.position - vehicle_pos))
            nearest_bs = min(network.base_stations, key=lambda bs: np.linalg.norm(bs.position - vehicle_pos))
            return self.luav_bs_time(vehicle_pos, nearest_upper_uav.position, nearest_bs.position, task,
                                     nearest_bs.compute_capacity, network.vehicles[task.vehicle_id].tx_power,
                                     nearest_upper_uav.tx_power)

        else:
            raise ValueError(f"Invalid offload target: {offload_target}")
