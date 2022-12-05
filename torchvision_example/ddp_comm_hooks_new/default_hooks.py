import threading
from typing import Any, Callable, List
import time
import torch
import torch.distributed as dist
from queue import Queue
import os

import struct
import fcntl
import socket
import psutil
import subprocess
import math
import numpy as np

from sklearn.cluster import SpectralClustering


os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

def _allreduce_fut(
    process_group: dist.ProcessGroup, tensor: torch.Tensor
) -> torch.futures.Future:
    group_to_use = process_group if process_group is not None else dist.group.WORLD
    "Averages the input gradient tensor by allreduce and returns a future."
    fut = dist.all_reduce(tensor, group=group_to_use,
                          async_op=True).get_future()
    # values_buffer = [torch.zeros_like(tensor, device=tensor.device) for _ in range(group_to_use.size())]
    # dist.all_gather(values_buffer, tensor, group=group_to_use, async_op=True)

    def div_by_group_size(fut):
        ret = [fut.value()[0].div_(group_to_use.size())]
        return ret

    return fut.then(div_by_group_size)

layers = {}
key_ef = 0
bucket_number = -1
model_name = 'resnet50_threshold'
num_step = 0

if os.path.exists('./4096_cifar_'+model_name+'_rank'+str(dist.get_rank())+'_localthreshold.txt'):
    os.remove('./4096_cifar_'+model_name+'_rank'+str(dist.get_rank())+'_localthreshold.txt')
if os.path.exists('./4096_cifar_'+model_name+'_rank'+str(dist.get_rank())+'_globalthreshold.txt'):
    os.remove('./4096_cifar_'+model_name+'_rank'+str(dist.get_rank())+'_globalthreshold.txt')

def allreduce_hook(
    process_group: dist.ProcessGroup, bucket: dist.GradBucket
) -> torch.futures.Future:
    """
    This DDP communication hook just calls ``allreduce`` using ``GradBucket``
    tensors. Once gradient tensors are aggregated across all workers, its ``then``
    callback takes the mean and returns the result. If user registers this hook,
    DDP results is expected to be same as the case where no hook was registered.
    Hence, this won't change behavior of DDP and user can use this as a reference
    or modify this hook to log useful information or any other purposes while
    unaffecting DDP behavior.

    Example::
        >>> ddp_model.register_comm_hook(process_group, allreduce_hook)
    """
    #print("bucket len:", bucket.get_tensor().nelement() * bucket.get_tensor().element_size(), "\ttype:", bucket.get_tensor().dtype)
    
    # global layers,bucket_number,key_ef
    # buctensor = bucket.get_tensor()

    # if not bucket_number == bucket.get_bucket_number():
    #     bucket_number = bucket.get_bucket_number()

    # if bucket.get_local_threshold() > 0:
    #     if layers.get(str(bucket.get_index())) == None:
    #         layers[str(bucket.get_index())] = torch.zeros(bucket.get_tensor().size()).to(buctensor.device)
    #     buctensor = bucket.get_tensor() + layers[str(bucket.get_index())].to(bucket.get_tensor().device)
    
    # return _allreduce_topk_threshold(process_group, buctensor, bucket.get_index(),bucket.get_local_threshold())
    
    
    # if key_ef >= bucket_number:
    #     if layers.get(str(bucket.get_index())) == None:
    #         layers[str(bucket.get_index())] = torch.zeros(bucket.get_tensor().size()).to(buctensor.device)
    #     buctensor = bucket.get_tensor() + layers[str(bucket.get_index())].to(bucket.get_tensor().device)
    # return _allreduce_topk(process_group, buctensor, bucket.get_index())
    # #return _allreduce_yangkh_hierachical_topk(process_group, buctensor, bucket.get_index())
    
    return _allreduce_fut(process_group, bucket.get_tensor())
    #return _allreduce_hierachical_new(process_group, bucket.get_tensor())

group1 = None  # 最顶层
# group21 = None  # 底层 组1
# group22 = None  # 底层 组2
# group23 = None  # 底层 组3
global_master = None #全局master
level2groups:list = None  # 第二层的组
level2masters:list = None  # 第二层的masters
lock = threading.Lock()
lock2 = threading.Lock()
in_bucket_count = 0
next_bucket_thread = 0
next_bucket_thread_lock2 = 0
my_events = []
q1 = Queue(maxsize=1)
q2 = Queue(maxsize=1)

def get_ip_address(ifname):
    if type(ifname) == str:
        ifname = bytes(ifname, encoding='ascii')
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    return socket.inet_ntoa(fcntl.ioctl(
        s.fileno(),
        0x8915,  # SIOCGIFADDR
        struct.pack('256s', ifname[:15])
    )[20:24])

def is_netserver_exists():
    for proc in psutil.process_iter():
        try:
            proc_name = proc.name()
            # proc_cmd_line = proc.cmdline()
            if "netserver" in proc_name:
                return True
        except psutil.NoSuchProcess:
            pass
    return False

def gather_ip(ip):
    # 不能用 [tensor] * world_size 因为这样只创建了1个tensor和多个引用
    ip_list = []
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tensor_list = [torch.zeros(4, dtype=torch.int64).to(
        device) for _ in range(dist.group.WORLD.size())]  # 提前分配好缓冲区
    tensor_ip = torch.tensor(ip, dtype=torch.int64).to(device)
    dist.all_gather(tensor_list, tensor_ip)
    for i in tensor_list:
        ip_tensor = i.tolist()
        ip_list.append(f"{ip_tensor[0]}.{ip_tensor[1]}.{ip_tensor[2]}.{ip_tensor[3]}")
    print("IP list", ip_list)
    return ip_list

def test_latency(dest, test_length_in_seconds=1):
    cmd = f"netperf -H {dest} -t TCP_RR -l {test_length_in_seconds} -- -o min_latency,max_latency,mean_latency "
    output = subprocess.check_output(cmd.split(' '))
    lines = output.decode().strip().split('\n')[-2:]
    return {k: v for k, v in zip(lines[0].split(','), lines[1].split(','))}

def init_my_group():
    nic_iface = os.environ.get("NCCL_SOCKET_IFNAME")
    if nic_iface is None:
        nic_iface = "eth0"

    ip1 = [int(i) for i in get_ip_address(nic_iface).split('.')]
    #print("the NIC is ", nic_iface, "; ip is:", ip1)
    if not is_netserver_exists():
        os.system("mkdir -p ~/logs; nohup netserver -p 49999 >>~/logs/output.log 2>&1 &")

    ip_list = gather_ip(ip1)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    latency = torch.zeros(dist.group.WORLD.size(),device = device)
    for i in range(dist.get_rank()+1,dist.get_rank()+1+math.ceil(dist.group.WORLD.size()/2)):
        print(ip_list[i%dist.group.WORLD.size()])
        latency[i%dist.group.WORLD.size()] = float(test_latency(ip_list[i%dist.group.WORLD.size()])["Mean Latency Microseconds"])

    latencies = [torch.zeros(dist.group.WORLD.size()).to(device) for _ in range(dist.group.WORLD.size())]
    dist.all_gather(latencies, latency)
    latencies_list = []
    for l in latencies:
        latencies_list.append(l.tolist())
    
    for i in range(0,dist.group.WORLD.size()):
        for j in range(0,dist.group.WORLD.size()):
            if latencies_list[i][j] == 0:
                latencies_list[i][j] = 20
                latencies_list[i][j] = latencies_list[j][i]
        #print('\t'.join(str(int(v)) for v in latencies_list[i]))
    
    obj = np.array(latencies_list)
    obj += obj.T
    obj = 1 / obj  # 元素除法
    obj = np.array(obj)

    # %% 聚类
    #print(obj)
    cluster_model = SpectralClustering(affinity='precomputed', n_clusters=int(math.sqrt(dist.group.WORLD.size())),assign_labels='discretize')
    #cluster_model.fit(obj)
    labels = cluster_model.fit(obj).labels_

    id_map = dict()
    for c in labels:
        if c in id_map:
            continue
        else:
            id_map[c] = len(id_map)
    for idx in range(len(labels)):
        labels[idx] = id_map[labels[idx]]
    # 恢复原来的延迟
    obj = 1 / obj
    obj *= -np.eye(dist.group.WORLD.size()) + 1
    lms = []
    # 计算组内主节点
    for l in range(0, int(math.sqrt(dist.group.WORLD.size()))):
        idx = np.where(labels == l)[0]
        arr = obj[idx][:, idx]
        lm = np.argmin(np.sum(arr, 0))
        lms.append(idx[lm])
    # 计算全局主节点
    arr = obj[lms][:, lms]
    gm = np.argmin(np.sum(arr, 0))
    gm = lms[gm]

    global group1, level2groups, level2masters, global_master
    level2groups = [None for _ in range(0,int(math.sqrt(dist.group.WORLD.size())))]
    for i in range(0,int(math.sqrt(dist.group.WORLD.size()))):
        level2groups[i] = dist.new_group(ranks=[idx for idx in range(len(labels)) if labels[idx] == i], backend=dist.Backend.NCCL)
        #print([idx for idx in range(len(labels)) if labels[idx] == i])
    #group1 = dist.new_group(ranks=lms, backend=dist.Backend.NCCL)
    #level2groups = [group21, group22]
    level2masters = lms
    group1 = dist.new_group(ranks=level2masters, backend=dist.Backend.NCCL)
    global_master = gm

    print("class:", labels)
    print("masters:", lms, gm)

    # if dist.group.WORLD.size() == 6:
    #     __init_groups_of_6_nodes()
    # elif dist.group.WORLD.size() == 10:
    #     __init_groups_of_10_nodes()
    # elif dist.group.WORLD.size() == 8:
    #     __init_groups_of_8_nodes()


# def __init_groups_of_6_nodes():
#     global group1, group21, group22, level2groups, level2masters
#     group21 = dist.new_group(ranks=[0, 1, 2], backend=dist.Backend.NCCL)
#     group22 = dist.new_group(ranks=[3, 4, 5], backend=dist.Backend.NCCL)
#     group1 = dist.new_group(ranks=[0, 3], backend=dist.Backend.NCCL)
#     level2groups = [group21, group22]
#     level2masters = [0, 3]

# def __init_groups_of_8_nodes():
#     global group1, group21, group22, level2groups, level2masters
#     group21 = dist.new_group(ranks=[0, 1, 2, 3], backend=dist.Backend.NCCL)
#     group22 = dist.new_group(ranks=[4, 5, 6, 7], backend=dist.Backend.NCCL)
#     group1 = dist.new_group(ranks=[0, 4], backend=dist.Backend.NCCL)
#     level2groups = [group21, group22]
#     level2masters = [0, 4]


# def __init_groups_of_10_nodes():
#     global group1, group21, group22, group23, level2groups, level2masters
#     group21 = dist.new_group(ranks=[0, 1, 2], backend=dist.Backend.NCCL)
#     group22 = dist.new_group(ranks=[3, 4, 5], backend=dist.Backend.NCCL)
#     group23 = dist.new_group(ranks=[6, 7, 8, 9], backend=dist.Backend.NCCL)
#     group1 = dist.new_group(ranks=[0, 3, 6], backend=dist.Backend.NCCL)
#     level2groups = [group21, group22, group23]
#     level2masters = [0, 3, 6]

def _allreduce_topk_threshold(process_group: dist.ProcessGroup, tensor: torch.Tensor, bucindex:int, threshold:float) -> torch.futures.Future:
    group_to_use = process_group if process_group is not None else dist.group.WORLD

    def store_small_gradients():
        global layers
        tensor[indices_buffer[torch.distributed.get_rank()].long()] = 0
        layers[str(bucindex)] = 0.1*tensor

    if threshold > 0:
        values = tensor[tensor.abs()>=threshold]
        indices = torch.range(0,tensor.numel()-1,device = tensor.device)[tensor.abs()>=threshold]
        length_buffer = [None for _ in range(group_to_use.size())]
        dist.all_gather_object(length_buffer, values.numel(), group=group_to_use)
        max_length = max(length_buffer)
        if values.numel() < max_length:
            values = torch.cat([values.to(tensor.device),torch.zeros([max_length - values.numel()], device=tensor.device)])
            indices = torch.cat([indices.to(tensor.device),torch.ones([max_length - indices.numel()], device=tensor.device)])

        indices_buffer = [torch.zeros_like(indices, device=tensor.device) for _ in range(group_to_use.size())]
        values_buffer = [torch.zeros_like(values, device=tensor.device) for _ in range(group_to_use.size())]
        fut1 = dist.all_gather(indices_buffer, indices.to(tensor.device), group=group_to_use, async_op=True).get_future()
        fut = dist.all_gather(values_buffer, values.to(tensor.device), group=group_to_use, async_op=True).get_future()

    else: 
        k = int(len(tensor) * 0.01)
        # start.record()
        result = torch.topk(tensor.abs(), k)
        # end.record()
        # torch.cuda.synchronize()
        #print('c',start.elapsed_time(end))
        indices_buffer = [torch.zeros_like(result.indices, device=tensor.device) for _ in range(group_to_use.size())]
        values_buffer = [torch.zeros_like(result.values, device=tensor.device) for _ in range(group_to_use.size())]
        fut1 = dist.all_gather(indices_buffer, result.indices.to(tensor.device), group=group_to_use, async_op=True).get_future()
        fut = dist.all_gather(values_buffer, tensor[result.indices], group=group_to_use, async_op=True).get_future()

    def div_by_group_size(fut):
        if threshold > 0:
            store_small_gradients()
        tensor.zero_()
        for i in range(group_to_use.size()):
            tensor.scatter_add_(0, indices_buffer[i].long(), values_buffer[i])
        return [tensor.div_(group_to_use.size())]

    return fut.then(div_by_group_size)

def _allreduce_topk(process_group: dist.ProcessGroup, tensor: torch.Tensor, bucindex:int) -> torch.futures.Future:
    group_to_use = process_group if process_group is not None else dist.group.WORLD
    k = int(len(tensor) * 0.01)
    result = torch.topk(tensor.abs(), k)
    indices_buffer = [torch.zeros_like(result.indices, device=tensor.device) for _ in range(group_to_use.size())]
    values_buffer = [torch.zeros_like(result.values, device=tensor.device) for _ in range(group_to_use.size())]
    fut = dist.all_gather(values_buffer, tensor[result.indices], group=group_to_use, async_op=True).get_future()
    fut1 = dist.all_gather(indices_buffer, result.indices.to(tensor.device), group=group_to_use, async_op=True).get_future()

    global layers,key_ef,bucket_number
    def store_small_gradients():
        global num_step
        tensor[indices_buffer[torch.distributed.get_rank()]] = 0
        layers[str(bucindex)] = tensor*(0.5+0.1*int(num_step/(bucket_number*100)))
        if num_step < bucket_number*500:
            num_step+=1

    def div_by_group_size(fut):
        global layers,key_ef,bucket_number
        if key_ef >= bucket_number:
            store_small_gradients()
        else:
            key_ef += 1
        tensor.zero_()
        for i in range(group_to_use.size()):
            tensor.scatter_add_(0, indices_buffer[i], values_buffer[i])
        return [tensor.div_(group_to_use.size())]

    return fut.then(div_by_group_size)


s = torch.cuda.Stream()
# t1 = [
#     torch.cuda.Stream(), torch.cuda.Stream(), torch.cuda.Stream(), torch.cuda.Stream(), 
#     torch.cuda.Stream(), torch.cuda.Stream(), torch.cuda.Stream(), torch.cuda.Stream(),
#     torch.cuda.Stream(), torch.cuda.Stream(), torch.cuda.Stream(), torch.cuda.Stream(),
#     torch.cuda.Stream(), torch.cuda.Stream(), torch.cuda.Stream(), torch.cuda.Stream(),
#     torch.cuda.Stream(), torch.cuda.Stream(), torch.cuda.Stream(), torch.cuda.Stream(),
#     torch.cuda.Stream(), torch.cuda.Stream(), torch.cuda.Stream(), torch.cuda.Stream(),
#     torch.cuda.Stream(), torch.cuda.Stream(), torch.cuda.Stream(), torch.cuda.Stream(),
#     torch.cuda.Stream(), torch.cuda.Stream(), torch.cuda.Stream(), torch.cuda.Stream(),
#     torch.cuda.Stream(), torch.cuda.Stream(), torch.cuda.Stream(), torch.cuda.Stream()
#     ]
t1:list = None

def _allreduce_yangkh_hierachical_topk(process_group: dist.ProcessGroup, tensor: torch.Tensor, bucindex:int) -> torch.futures.Future:
    """
    10个物理GPU的情况，rank 分为3组 0~2 3~5 6~9

    """
    global group1, level2groups, level2masters,global_master,t1,layers,key_ef,bucket_number,in_bucket_count
    group_to_use = process_group if process_group is not None else dist.group.WORLD
    if t1 == None:
        t1 = [torch.cuda.Stream() for _ in range(bucket_number)]
    elif len(t1) < bucket_number:
        t1.extend([torch.cuda.Stream() for _ in range(bucket_number-len(t1))])
    thread_id = in_bucket_count
    in_bucket_count += 1
    k = int(tensor.nelement() * 0.01)
    tensor.div_(group_to_use.size())  # 
    fut = torch.futures.Future()  # future任务
    # print("bucket len:", tensor.nelement() * tensor.element_size(), "\ttype:", tensor.dtype)
    rank = dist.get_rank(group_to_use)
    me = torch.cuda.Event()
    
    torch.cuda.current_stream().record_event(me)

    def my_thread(fut: torch.futures.Future, tensor: torch.Tensor):
        global next_bucket_thread, next_bucket_thread_lock2
        global layers,key_ef,bucket_number
        if thread_id % bucket_number == bucket_number-1:
            for _ in range(bucket_number):
                q1.put(None, block=True)
        q1.get(block=True)

        temp = tensor.clone()
        #temp1 = torch.zeros_like(tensor)

        LOG_LOCATION = 0
        with torch.cuda.stream(t1[thread_id % bucket_number]):
            t1[thread_id % bucket_number].wait_event(me)
            LOG_LOCATION = 1
            # print("本地计算TOP-K", tensor.sum(), k)
            result = torch.topk(tensor.abs(), k)  # 在GPU上计算全部元素的 topk
            LOG_LOCATION = 2
            values1 = tensor[result.indices]  # 取出topk个元素的值
            LOG_LOCATION = 3
            indices1 = result.indices  # topk个值的索引
            if key_ef >= bucket_number:
                #LOG_LOCATION = 171
                temp[indices1] = 0
                #LOG_LOCATION = 172
                #temp1+=temp
            te = torch.cuda.Event()
            t1[thread_id % bucket_number].record_event(te)

        while True:
            lock.acquire()
            if next_bucket_thread == thread_id: break
            lock.release()
        s.wait_event(te)
        with torch.cuda.stream(s):
            try:
                LOG_LOCATION = 4
                # 这里是第一轮 validx1 变量中的 1 表示的层, 1是最底层
                # validx1 = torch.cat((values1.float(), indices1.float()), 0)  # 值和索引合并到一起传输
                local_group_size = 0
                for g in level2groups:
                    if dist.get_rank(g) != -1: 
                        local_group_size = g.size()
                        break

                LOG_LOCATION = 5
                # validx_buffer = [torch.zeros_like(validx1) for _ in range(local_group_size)] if dist.get_rank(group1) != -1 else None
                value_buffer = [torch.zeros_like(values1) for _ in range(local_group_size)]
                LOG_LOCATION = 6
                indice_buffer = [torch.zeros_like(indices1) for _ in range(local_group_size)]
                LOG_LOCATION = 7
                for i in range(len(level2masters)):
                    #print('1')
                    dist.gather(values1, value_buffer if rank==level2masters[i] else None, dst=level2masters[i], group=level2groups[i], async_op=False)
                    #print('2')
                    dist.gather(indices1, indice_buffer if rank==level2masters[i] else None, dst=level2masters[i], group=level2groups[i], async_op=False)
                    #print('3')
                    # print("buffer | len:", len(validx_buffer), "\tnele:", validx1.nelement(), "\tdtype:", validx1.dtype)
                    #dist.all_gather(value_buffer, values1, group=level2groups[i], async_op=False)
                    #dist.all_gather(indice_buffer, indices1, group=level2groups[i], async_op=False)
                LOG_LOCATION = 8
                if dist.get_rank(group1) != -1:
                    sparse_tensor = None
                    for i in range(local_group_size):
                        indices = indice_buffer[i]
                        values = value_buffer[i]
                        LOG_LOCATION = 9
                        if i == 0:
                            LOG_LOCATION = 91
                            sparse_tensor = torch.sparse_coo_tensor(indices.reshape(1, -1), values, [tensor.nelement()])
                        else:
                            LOG_LOCATION = 92
                            sparse_tensor.add_(torch.sparse_coo_tensor(indices.reshape(1, -1), values, [tensor.nelement()]))
                        LOG_LOCATION = 10
                    
                    LOG_LOCATION = 11
                    sparse_tensor = sparse_tensor.coalesce()  # 收缩 (索引相同的要进行合并)
                    LOG_LOCATION = 12
                    sparse_tensor_values = sparse_tensor._values()
                    LOG_LOCATION = 13
                    sparse_tensor_indices = sparse_tensor._indices()
                    LOG_LOCATION = 14
                    result = torch.topk(sparse_tensor_values.abs(), k)  # 对值最大的k个取索引
                    LOG_LOCATION = 15
                    values2 = sparse_tensor_values[result.indices]
                    LOG_LOCATION = 16
                    indices2 = sparse_tensor_indices[0][result.indices]
                    LOG_LOCATION = 17
                    
                    if key_ef >= bucket_number:
                        LOG_LOCATION = 171
                        #sparse_tensor_values[indices2] = 0
                        LOG_LOCATION = 173
                        temp.scatter_add_(dim=0, index=sparse_tensor_indices[0], src=sparse_tensor_values)
                        temp.scatter_add_(dim=0, index=indices2, src=-1*values2)
                else:
                    values2 = torch.zeros_like(values1)
                    indices2 = torch.zeros_like(indices1)
                # 传输 values2 和 indices2
                # validx2 = torch.cat((values2.float(), indices2.float()), 0)
                LOG_LOCATION = 18
                local_group_size = 0 if dist.get_rank(group1) == -1 else group1.size()
                # validx_buffer = [torch.zeros_like(validx2) for _ in range(local_group_size)] if rank == 0 else None
                value_buffer = [torch.zeros_like(values2) for _ in range(local_group_size)]
                indice_buffer = [torch.zeros_like(indices2) for _ in range(local_group_size)]
                LOG_LOCATION = 19
                #print('4')
                dist.gather(values2, value_buffer if rank == global_master else None, dst=global_master, group=group1)  # 2021年6月24日15:07:01出错了
                #print('5')
                dist.gather(indices2, indice_buffer if rank == global_master else None, dst=global_master, group=group1)  # 2021年6月24日15:07:01出错了
                #print('6')
                #dist.all_gather(value_buffer, values2, group=group1)  # 2021年6月24日15:07:01出错了
                #dist.all_gather(indice_buffer, indices2, group=group1)
                LOG_LOCATION = 20
                if rank == global_master:  # global master
                    sparse_tensor1 = None
                    for i in range(local_group_size):
                        # values, indices = validx_buffer[i].chunk(2)
                        indices1 = indice_buffer[i]
                        values1 = value_buffer[i]
                        # print('validx_buffer[i].device', validx_buffer[i].device, '\n', 'indices.device', indices.device)
                        if i == 0:
                            sparse_tensor1 = torch.sparse_coo_tensor(indices1.reshape(1, -1), values1, [tensor.nelement()])
                        else:
                            sparse_tensor1.add_(torch.sparse_coo_tensor(indices1.reshape(1, -1), values1, [tensor.nelement()]))
                    sparse_tensor1 = sparse_tensor1.coalesce()
                    sparse_tensor_values1 = sparse_tensor1._values()
                    sparse_tensor_indices1 = sparse_tensor1._indices()
                    result1 = torch.topk(sparse_tensor_values1.abs(), k)
                    values3 = sparse_tensor_values1[result1.indices].to(tensor.device)
                    indices3 = sparse_tensor_indices1[0][result1.indices].to(tensor.device)
                    if key_ef >= bucket_number:
                        # LOG_LOCATION = 201
                        # sparse_tensor_values1[indices3] = 0
                        # LOG_LOCATION = 202
                        #temp[sparse_tensor_indices[0]] = 0
                        LOG_LOCATION = 203
                        temp.scatter_add_(dim=0, index=sparse_tensor_indices1[0], src=sparse_tensor_values1)
                        temp.scatter_add_(dim=0, index=indices3, src=-1*values3)
                else:
                    values3 = torch.zeros_like(values1, device=tensor.device)
                    indices3 = torch.zeros_like(indices1, device=tensor.device)
            except Exception as e:
                print('[位置: %s]' % LOG_LOCATION, 'thread', thread_id, '打印异常:', repr(e))
                exit(0)
            finally:
                #print('7')
                next_bucket_thread += 1
                lock.release()
            LOG_LOCATION = 21
                # print("释放锁 id=", thread_id, "修改 next_bucket_thread 为", next_bucket_thread, time.time())

            if thread_id % bucket_number == bucket_number-1:
                for _ in range(bucket_number):
                    q2.put(None, block=True)

            q2.get(block=True)
        while True:
            lock2.acquire()
            if next_bucket_thread_lock2 == thread_id: break
            lock2.release()
        try:
            # move to 
            #print('8')
            dist.broadcast(values3, global_master, group=group_to_use)  # NCCL
            dist.broadcast(indices3, global_master, group=group_to_use)
            tensor.zero_()
            tensor.scatter_add_(dim=0, index=indices3, src=values3)
            # print(f'{thread_id}:', torch.max(tensor))
            #print('9')
            if key_ef >= bucket_number:
                global num_step
                # zero_indices = torch.range(0,temp.numel()-1,device = temp.device)[temp==0]
                # temp -= layers[str(bucindex)].to(temp.device)
                # temp[zero_indices.long()] = 0
                # layers[str(bucindex)] = temp*(0.5+0.1*int(num_step/(bucket_number*1000)))
                layers[str(bucindex)] = temp*(0.5+0.1*int(num_step/(bucket_number*100)))
                if num_step < bucket_number*500:
                    num_step+=1
            else:
                key_ef += 1
                

        except Exception as e:
            print("打印异常:", repr(e))
        finally:
            next_bucket_thread_lock2 += 1
            lock2.release()
            fut.set_result([tensor])
        
    t = threading.Thread(
        target=my_thread,
        args=(fut, tensor)
    )
    t.start()
    return fut


def finish(log_path='gloo-events-%s-0.log'):
    """
    将生成的日志导出；关闭通信进程
    """
    import socket
    with open(log_path % socket.gethostname(), 'w') as f:
        for event in my_events:
            # print(event)
            f.write(','.join([str(e) for e in event]))
            f.write('\n')


def _allreduce_hierachical(process_group: dist.ProcessGroup, tensor: torch.Tensor) -> torch.futures.Future:
    """
    10个物理GPU的情况，rank 分为3组 0~2 3~5 6~9

    """
    global group1, level2groups, level2masters,global_master,t1,bucket_number,in_bucket_count,key_ef
    group_to_use = process_group if process_group is not None else dist.group.WORLD
    
    #fut = torch.futures.Future()

    #fut=[None for _ in range(len(level2masters))]

    fut = None

    result = torch.futures.Future()

    for i in range(len(level2masters)):
        if dist.get_rank() == level2masters[i]:   
            fut = dist.reduce(tensor, dst=level2masters[i],group=level2groups[i],async_op=True).get_future()
        else:
            dist.reduce(tensor, dst=level2masters[i],group=level2groups[i],async_op=True)
    
    
        #fut[i] = dist.all_reduce(tensor, group=level2groups[i],
        #                  async_op=True).get_future()
    
    def callback1(fut):
        fut1 = dist.reduce(tensor, dst=global_master,group=group1,async_op=True).get_future()
        def callback2(fut):
            global key_ef
            dist.broadcast(tensor.div_(group_to_use.size()), global_master, group=group_to_use)
            result.set_result([tensor])
            #if key_ef < 10:
            print('3:')
        fut1.then(callback2)

    if fut is not None:
        fut.then(callback1)
        
        #if key_ef < 10:
        print('1:')
        key_ef += 1
        #result.set_result([tensor])
        #if key_ef < 10:
        print('2')
        return result
    else:
        dist.broadcast(tensor.div_(group_to_use.size()), global_master, group=group_to_use)
        result.set_result([tensor])
        #if key_ef < 10:
        print('1:')
        key_ef += 1
        return result
    # fut[len(level2masters)-1].then(callback)
    #fut = dist.reduce(tensor, dst=global_master, group=group1,async_op=True).get_future()

    # def callback2(fut):
    #     dist.broadcast(tensor.div_(group_to_use.size()), global_master, group=group_to_use)
    #     return [tensor]
    # def callback(fut):
    #     dist.broadcast(tensor.div_(group_to_use.size()), global_master, group=group_to_use)
    #     return [tensor]

    

def _allreduce_hierachical_new(process_group: dist.ProcessGroup, tensor: torch.Tensor) -> torch.futures.Future:
    global group1, level2groups, level2masters,global_master,t1,layers,key_ef,bucket_number,in_bucket_count
    group_to_use = process_group if process_group is not None else dist.group.WORLD
    if t1 == None:
        t1 = [torch.cuda.Stream() for _ in range(bucket_number)]
    elif len(t1) < bucket_number:
        t1.extend([torch.cuda.Stream() for _ in range(bucket_number-len(t1))])
    thread_id = in_bucket_count
    in_bucket_count += 1
    fut = torch.futures.Future()  # future任务
    rank = dist.get_rank(group_to_use)

    def my_thread(fut: torch.futures.Future, tensor: torch.Tensor):
        global next_bucket_thread, next_bucket_thread_lock2
        global layers,key_ef,bucket_number

        # cpu barrier
        if thread_id % bucket_number == bucket_number-1:
            for _ in range(bucket_number):
                q1.put(None, block=True)
        q1.get(block=True)

        while True:
            lock.acquire()
            if next_bucket_thread == thread_id: break
            lock.release()
        with torch.cuda.stream(s):
            try:
                local_group_size = 0
                for g in level2groups:
                    if dist.get_rank(g) != -1: 
                        local_group_size = g.size()
                        break
                # if key_ef < 1 and thread_id % bucket_number == 1:
                #     print('1:',tensor[0])

                for i in range(len(level2masters)):
                    dist.reduce(tensor, dst=level2masters[i],group=level2groups[i])                                       
                #if key_ef < 1 and thread_id % bucket_number == 1:
                torch.cuda.synchronize()
                dist.reduce(tensor, dst=global_master, group=group1)

                # if dist.get_rank() == global_master:
                #     values = tensor.div_(group_to_use.size())
                #     if key_ef < 1 and thread_id % bucket_number == 1:
                # print('3:',values[0])
                # else:
                #     values = torch.zeros_like(tensor, device=tensor.device)
            except Exception as e:
                print('thread', thread_id, '打印异常:', repr(e))
                exit(0)
            finally:
                next_bucket_thread += 1
                lock.release()

            # cpu barrier
            if thread_id % bucket_number == bucket_number-1:
                for _ in range(bucket_number):
                    q2.put(None, block=True)
            q2.get(block=True)
        
        while True:
            lock2.acquire()
            if next_bucket_thread_lock2 == thread_id: break
            lock2.release()
        try:
            
            dist.broadcast(tensor.div_(group_to_use.size()), global_master, group=group_to_use)  # NCCL

            if key_ef < 1 and thread_id % bucket_number == 1:
                print('4:',tensor[0])
                key_ef = 1
            
        except Exception as e:
            print("打印异常:", repr(e))
        finally:
            next_bucket_thread_lock2 += 1
            lock2.release()
            fut.set_result([tensor])
    t = threading.Thread(
        target=my_thread,
        args=(fut, tensor)
    )
    t.start()
    return fut