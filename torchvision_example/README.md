# torchvision官方的imagenet训练代码

传入了max_worker参数（n为64），因此可以在少于64个gpu上进行训练，以8个gpu为例，训练将在每8步的前7步不更新参数并存储对应梯度，并在最后一步和其他节点（也是一样的过程）通信同步梯度结果并更新参数。

需要修改数据划分代码（对应torch.utils.data相关内容）和训练流程代码（对应torch.nn.parallel.DistributedDataParallel代码，其中主要代码在reducer.cpp中）
