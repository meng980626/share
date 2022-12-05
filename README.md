# share

## Setup the environment
To install Pytorch: 

`conda create -n env_name python=3.9`

`conda activate env_name`

`pip install torch==1.9.0+cu112 torchvision==0.10.0+cu112 torchaudio==0.9.0 -f https://download.pytorch.org/whl/torch_stable.html`

Prepare for compiling Pytorch:

`conda install astunparse numpy ninja pyyaml mkl mkl-include setuptools cmake cffi typing_extensions future six requests dataclasses`

Compile Pytorch:

`pip uninstall torch`

`export CMAKE_PREFIX_PATH=${CONDA_PREFIX:-"$(dirname $(which conda))/../"}`

`cd $SourceCodePath/pytorch`

`python setup.py install`

## 介绍
本项目用于使数据并行的n个worker可以在少于n的gpu上共享执行，基本思路是使多个worker在一个gpu上轮流执行前向反向并在最后一个worker发起同步并平均所有梯度然后再更新，具体原理可见easyscale论文（模仿其思路实现）。由于目前只是demo，n默认为2的整数次幂，gpu数量也是2的整数次幂且少于n，并且默认是同构集群。使用时需要在用户训练代码中对torch.nn.parallel.DistributedDataParallel, torch.utils.data.distributed.DistributedSampler, torch.utils.data.DataLoader三个类添加参数max_worker=n即可，测试代码见/torchvision_example。
