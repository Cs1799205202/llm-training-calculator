from pynvml.nvml import nvmlPciInfo_t
import torch
import torch.distributed as dist
import random
import subprocess
import os
from pathlib import Path
import argparse
import shutil
import platform
from pynvml import (
    nvmlInit,
    nvmlDeviceResetGpuLockedClocks,
    nvmlShutdown,
    nvmlDeviceGetHandleByIndex,
    nvmlDeviceSetGpuLockedClocks,
    nvmlDeviceGetPciInfo,
)


def get_nvml_gpu_id(torch_gpu_id):
    """
    Remap torch device id to nvml device id, respecting CUDA_VISIBLE_DEVICES.

    If the latter isn't set return the same id
    """
    # if CUDA_VISIBLE_DEVICES is used automagically remap the id since pynvml ignores this env var
    if "CUDA_VISIBLE_DEVICES" in os.environ:
        ids = list(map(int, os.environ.get("CUDA_VISIBLE_DEVICES", "").split(",")))
        return ids[torch_gpu_id]  # remap
    else:
        return torch_gpu_id


def setup_distributed():
    dist.init_process_group()
    local_rank = dist.get_rank()
    nnodes = dist.get_world_size()
    return local_rank, nnodes


def check_success(abnormal_content, gpu_rank_map_file_content, node, gpu_id):
    print("Checking...")
    print(abnormal_content)
    print(gpu_rank_map_file_content)
    print(node)
    print(gpu_id)

    # D\tP\tT\tGPU\tNODE
    l = map(
        lambda x: x.split("\t"),
        gpu_rank_map_file_content.strip().split("\n")[1:],
    )
    abnormal_rank = abnormal_content.strip().split(" ")[1].split("-")
    for i in l:
        if i[0] == abnormal_rank[0] and i[1] == abnormal_rank[1] and i[2] == abnormal_rank[2]:
            # print(i[4], node, i[3], gpu_id)
            gpu_id = str(gpu_id)
            print(f'Checking: "{i[4]}" == "{node}" and "{i[3]}" == "{gpu_id}"')
            print(type(i[4]), type(node), type(i[3]), type(gpu_id))
            if i[4] == node and i[3] == gpu_id:
                print("True")
                return True
            else:
                print("False")
                return False


def run_benchmark_round_gpu_clock(round_number, rank, nnodes, gpu_per_node, log_file):
    world_size = nnodes * gpu_per_node

    nvmlInit()
    for i in range(gpu_per_node):
        index = get_nvml_gpu_id(i)
        handle = nvmlDeviceGetHandleByIndex(index)
        nvmlDeviceResetGpuLockedClocks(handle)
    nvmlShutdown()

    # 每个 rank 生成一个随机数
    local_random_number = random.random()
    random_numbers = [torch.tensor(0.0) for _ in range(nnodes)]
    dist.all_gather(random_numbers, torch.tensor(local_random_number))

    # 找出最大随机数的 rank
    max_rank = torch.argmax(torch.tensor(random_numbers)).item()
    print(max_rank)

    # 如果是最大 rank，选择一个 GPU 进行降频
    if rank == max_rank:
        gpu_id = random.randint(0, gpu_per_node - 1)
        nvmlInit()
        index = get_nvml_gpu_id(gpu_id)
        handle = nvmlDeviceGetHandleByIndex(index)
        nvmlDeviceSetGpuLockedClocks(handle, 0, 900)
        nvmlShutdown()

        with open(log_file, "a") as f:
            f.write(f"Round {round_number}:\n")
            f.write(f"Throttled rank: {max_rank}\n")
            f.write(f"Node: {platform.node()}\n")
            f.write(f"Throttled GPU: {gpu_id}\n")
        
        if max_rank == 0:
            node, gpu_id = platform.node(), gpu_id
        else:
            dist.send_object_list([platform.node(), gpu_id], 0)
    elif rank == 0:
        objects = [None, None]
        dist.recv_object_list(objects, max_rank)
        node, gpu_id = objects
        

    # 同步各个 rank
    dist.barrier()

    env = os.environ.copy()

    # 执行训练脚本
    process = subprocess.Popen(
        "bash script/benchmark.sh train",
        shell=True,
        env=env,
    )
    try:
        process.wait(timeout=240)
    except subprocess.TimeoutExpired:
        with open(log_file, "a") as f:
            f.write("Timeout occurred\n")
            f.write("Skipping aggregation\n")
        process.kill()

    # 如果是 rank 0，执行汇总和文件处理
    if rank == 0:
        with open("abnormal.txt", "w") as f:
            f.truncate(0)

        p = subprocess.run("python script/aggregate.py -d", shell=True)

        Path(Path.cwd() / "experiment_gpu_clock" / str(round_number)).mkdir(
            parents=True, exist_ok=True
        )
        benchmark_file = Path("benchmark.json")
        benchmark_file.rename(
            Path.cwd() / "experiment_gpu_clock" / str(round_number) / "benchmark.json"
        )

        intermediate_file = (Path.cwd() / "Megatron").glob("benchmark-data-*.json")
        for file in intermediate_file:
            file.rename(
                Path.cwd() / "experiment_gpu_clock" / str(round_number) / file.name
            )

        with open("abnormal.txt", "r") as f:
            abnormal_content = f.read()
        
        with open("Megatron/gpu-rank-map.txt", "r") as f:
            gpu_rank_map = f.read()

        with open(log_file, "a") as f:
            f.write(f"Exit code: {p.returncode}\n")
            f.write(f"abnormal.txt content:\n{abnormal_content}\n")
            f.write(f"gpu-rank-map.txt content:\n{gpu_rank_map}\n")
            if check_success(abnormal_content, gpu_rank_map, node, gpu_id):
                f.write("Success\n")
            else:
                f.write("Failed\n")

            f.write("\n")

    dist.barrier()


def run_benchmark_round_gpu_pcie(round_number, rank, nnodes, gpu_per_node, log_file):
    world_size = nnodes * gpu_per_node

    pcie_bus_id = []
    nvmlInit()
    print(platform.node(), rank, nnodes)
    for i in range(gpu_per_node):
        index = get_nvml_gpu_id(i)
        handle = nvmlDeviceGetHandleByIndex(index)
        # nvmlDeviceResetGpuLockedClocks(handle)
        info: nvmlPciInfo_t = nvmlDeviceGetPciInfo(handle)
        bus = info.busIdLegacy.decode().lower()
        pcie_bus_id.append(bus)
        print(f'GPU {i} bus: "{bus}" {info}')
        shell_process = subprocess.run(
            "bash ./pcie_set_speed.sh " +  bus + " 4",
            shell=True,
            capture_output=True,
        )

    nvmlShutdown()

    # 每个 rank 生成一个随机数
    local_random_number = random.random()
    random_numbers = [torch.tensor(0.0) for _ in range(nnodes)]
    dist.all_gather(random_numbers, torch.tensor(local_random_number))

    # 找出最大随机数的 rank
    max_rank = torch.argmax(torch.tensor(random_numbers)).item()
    print(max_rank)

    # 如果是最大 rank，选择一个 GPU 进行降频
    if rank == max_rank:
        gpu_id = random.randint(0, gpu_per_node - 1)

        print(pcie_bus_id[gpu_id])
        shell_process = subprocess.run(
            "bash ./pcie_set_speed.sh " +  pcie_bus_id[gpu_id] + " 1",
            shell=True,
            capture_output=True,
        )
        print("hhhh",
            shell_process.args,
            shell_process.stdout.decode())

        with open(log_file, "a") as f:
            f.write(f"Round {round_number}:\n")
            f.write(f"Throttled rank: {max_rank}\n")
            f.write(f"Node: {platform.node()}\n")
            f.write(f"Throttled GPU: {gpu_id}\n")
        
        if max_rank == 0:
            node, gpu_id = platform.node(), gpu_id
        else:
            dist.send_object_list([platform.node(), gpu_id], 0)
    elif rank == 0:
        objects = [None, None]
        dist.recv_object_list(objects, max_rank)
        node, gpu_id = objects
        

    # 同步各个 rank
    dist.barrier()

    env = os.environ.copy()

    # 执行训练脚本
    process = subprocess.Popen(
        "bash script/benchmark.sh train",
        shell=True,
        env=env,
    )
    try:
        process.wait(timeout=240)
    except subprocess.TimeoutExpired:
        with open(log_file, "a") as f:
            f.write("Timeout occurred\n")
            f.write("Skipping aggregation\n")
        process.kill()

    # 如果是 rank 0，执行汇总和文件处理
    if rank == 0:
        with open("abnormal.txt", "w") as f:
            f.truncate(0)

        p = subprocess.run("python script/aggregate.py -d", shell=True)

        Path(Path.cwd() / "experiment_gpu_pcie" / str(round_number)).mkdir(
            parents=True, exist_ok=True
        )
        benchmark_file = Path("benchmark.json")
        benchmark_file.rename(
            Path.cwd() / "experiment_gpu_pcie" / str(round_number) / "benchmark.json"
        )

        intermediate_file = (Path.cwd() / "Megatron").glob("benchmark-data-*.json")
        for file in intermediate_file:
            file.rename(
                Path.cwd() / "experiment_gpu_pcie" / str(round_number) / file.name
            )

        with open("abnormal.txt", "r") as f:
            abnormal_content = f.read()
        
        with open("Megatron/gpu-rank-map.txt", "r") as f:
            gpu_rank_map = f.read()

        with open(log_file, "a") as f:
            f.write(f"Exit code: {p.returncode}\n")
            f.write(f"abnormal.txt content:\n{abnormal_content}\n")
            f.write(f"gpu-rank-map.txt content:\n{gpu_rank_map}\n")
            if check_success(abnormal_content, gpu_rank_map, node, gpu_id):
                f.write("Success\n")
            else:
                f.write("Failed\n")

            f.write("\n")

    dist.barrier()


def main():
    # 设置日志文件
    log_file = "experiment_log.txt"
    if os.path.exists(log_file):
        os.remove(log_file)

    # 初始化分布式进程组
    rank, world_size = setup_distributed()

    os.environ["CUDA_VISIBLE_DEVICES"] = "2,3,4,5"

    if rank == 0:
        dir = Path.cwd() / "experiment_gpu_clock"
        if dir.exists():
            shutil.rmtree(dir)
        dir = Path.cwd() / "experiment_gpu_pcie"
        if dir.exists():
            shutil.rmtree(dir)

    parser = argparse.ArgumentParser()

    parser.add_argument("--num_rounds", type=int, default=60)
    parser.add_argument("--gpu_clock", default=False, action="store_true") # done
    parser.add_argument("--gmem_clock", default=False, action="store_true")
    parser.add_argument("--gpu_pcie", default=False, action="store_true") # done
    parser.add_argument("--nic", default=False, action="store_true")
    parser.add_argument("--nic_pcie", default=False, action="store_true")
    
    parser.add_argument("--gpu_per_node", type=int, default=4)

    args = parser.parse_args()

    num_rounds = args.num_rounds

    if args.gpu_clock:

        for round_number in range(1, num_rounds + 1):
            run_benchmark_round_gpu_clock(
                round_number, rank, world_size, args.gpu_per_node, log_file
            )
    
    elif args.gpu_pcie:
        for round_number in range(1, num_rounds + 1):
            run_benchmark_round_gpu_pcie(
                round_number, rank, world_size, args.gpu_per_node, log_file
            )

    # 清理分布式进程组
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
