# simulations/test_vehicle.py
from node.vehicle import Vehicle
import numpy as np


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


def test_vehicle_nodes():
    # 创建 5 辆车辆
    vehicles = [Vehicle(vid=i) for i in range(5)]

    # 记录初始位置，确保后续位置不变
    initial_positions = {v.vid: v.position.copy() for v in vehicles}

    timeslot_count = 5  # 模拟 5 个时隙
    for t in range(timeslot_count):
        print(f"\n===== 时隙 {t} =====")
        for v in vehicles:
            # 每个时隙刷新任务：先清空任务队列
            v.task_queue.clear()
            # 生成新任务（用当前时隙作为 timestamp）
            task = v.generate_task(timestamp=float(t))
            # 打印车辆的基本状态信息
            state = v.get_state()
            print(f"车辆 {v.vid} 状态:")
            print(f"  位置             : {state['position']}")
            print(f"  速度             : {state['speed']:.2f}")
            print(f"  计算资源         : {state['self']:.2f} GHz")
            print(f"  当前任务队列     : {state['task_queue']}")
            # 打印任务详细信息
            print_task_details(v.current_task)
            # 检查车辆位置是否保持不变
            assert np.array_equal(v.position, initial_positions[v.vid]), f"车辆 {v.vid} 的位置发生变化！"
        print("=" * 40)


if __name__ == '__main__':
    test_vehicle_nodes()
