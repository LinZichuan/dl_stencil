# dl_stencil
## 任务背景
* 随着深度学习网络越来越复杂以及数据规模越来越大，训练过程中的计算量也在急剧增加，面对庞大的数据量，传统的CPU已经无法承载，用GPU进行深度学习训练已经成为必然的趋势。
* 目前，已经有很多机器学习框架，比如caffe、torch7、theano、mxnet等，这些框架设计好了底层的训练结构，提供给用户方便的训练参数设置和算法设计的接口，但是在底层的GPU加速上却做的有所不足。为此，NVIDIA推出了cuDNN，更好地利用GPU的特性，加速了大多数DNN中layer的计算速度。

## 所要解决的具体问题
* 在NVIDIA推出的cuDNN库中，进行了包括convolution、pooling、activation、softmax、tensor transformation等layer计算的加速，但是存在的一个问题是每一层的计算都是通过单个API调用来实现的，多层的计算就需要通过多次调用API来实现。如果可以在一个API调用中实现多层的计算，在层与层之间做内部的优化，也许效率还会有上升的空间。
* 在cuDNN的convolution等layer中，并行计算是通过把input features和kernels展开成冗余矩阵的形式，进而转化成矩阵乘矩阵的运算来实现的。这种方法虽然运算速度快，但是在内存的消耗上却是baseline的几倍。如果本实验的方法可以做到降低内存的消耗，那DNN将可以承载更大的数据和参数，更便于扩展。

## 相关技术
* GPU的架构了解
* cuda编程方法
* 深度学习训练过程(forward、backward)
* layer的计算方法以及优化方法

## 任务目标
* 保证计算结果正确性的前提下，在GPU上对DNN做层与层之间的计算优化
* 复现某一个深度学习网络，将本文的方法与cuDNN的多次单层调用的方法做对比

## 工作计划
* 1～2周：完成DNN的convoluiton/pooling/softmax的GPU baseline
* 3～5周：深入调研GPU的优化方法(shared memory、register、warp...)，对baseline进行一系列优化，并与cuDNN的单层调用做对比
* 6～9周：在层与层之间做优化，并与多层的cuDNN调用在时间效率和内存使用上作对比
* 10～14周：复现某一个深度学习网络，将本文的方法用于实际的深度学习训练过程中，与一些机器学习框架(比如caffe、mxnet等)的效率作对比
* 15～16周：撰写论文、后期答辩
