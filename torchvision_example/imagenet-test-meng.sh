scp root@10.11.108.120:/home/torchvision_examples/main.py /home/torchvision_examples
scp -r root@10.11.108.120:/home/torchvision_examples/ddp_comm_hooks_new /home/torchvision_examples


if ! [[ $(nvidia-smi) =~ "No running processes found" ]]; then
  echo "有其它程序在使用GPU, 程序退出, 相关进程如下:"
  nvi=$(nvidia-smi)
  res=$(grep -n '+-----------------------------------------------------------------------------+' <<< "$nvi" | awk -F: '{print $1}' | tail -n2)
  lineBegin=$(head -n1 <<< "$res")
  lineEnd=$(tail -n1 <<< "$res")
  processes=$(head -n $((lineEnd-1))<<< "$nvi" | tail -n $((lineEnd-lineBegin-5)))
  ps f $(awk '{print $5}' <<<"$processes")
  exit 1    # 使用source执行脚本的时候不能用exit命令 否则会让执行source的程序退出
fi

source /root/anaconda3/bin/activate  /root/anaconda3/envs/pytorch_5

declare master_pids
declare rank=$NODE_RANK

#echo "(" cd /home/torchvision_examples && NCCL_SOCKET_IFNAME="eth0" GLOO_SOCKET_IFNAME="eth0" python main.py -a resnet101 --dist-url 'tcp://10.10.108.4:12345' --dist-backend 'nccl' --multiprocessing-distributed --world-size 8 --rank ${rank} /mnt/imagenet-object-localization-challenge/ILSVRC/Data/CLS-LOC ")"
#( cd /home/pytorch_simple_classification_baselines && NCCL_SOCKET_IFNAME="eth0" GLOO_SOCKET_IFNAME="eth0" python ddp-test-meng.py ${rank} 8 "10.10.108.4:23579" 0 ) &
( cd /home/torchvision_examples && NCCL_SOCKET_IFNAME="eth0" GLOO_SOCKET_IFNAME="eth0" python main.py -a resnet101 --dist-url 'tcp://10.11.108.120:23579' --dist-backend 'nccl' --multiprocessing-distributed --world-size 8 --rank ${rank} /mnt/imagenet-object-localization-challenge/ILSVRC/Data/CLS-LOC ) &
master_pids=$!

# 开始接收SIGINT信号 Ctrl+C触发
handleSigInt() {
  echo "正在杀死所有进程"
  ps aux|grep -E "main.py"|grep -v grep|awk '{print $2}'|xargs kill -9
  exit 0  # 使用source执行脚本的时候不能用exit命令 否则会让执行source的程序退出
}

trap handleSigInt SIGINT

wait ${master_pids}
