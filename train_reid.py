import os
import argparse

import mindspore as ms

# from mindspore import context
# import mindspore.dataset as ds
import mindspore.dataset.transforms.c_transforms as C
import mindspore.dataset.vision.c_transforms as CV
from mindspore.dataset.vision import Inter
# from mindspore import dtype as mstype

# import mindspore.nn as nn
from mindspore.common.initializer import Normal

from mindspore.nn import Accuracy
from mindspore.train.callback import LossMonitor
# from mindspore import Model

from data.datasets import init_dataset, ImageDataset, ImageDatasetTrain

import numpy as np

np.random.seed(58)


# dataset_generator = DatasetGenerator()
# dataset = ms.dataset.GeneratorDataset(dataset_generator, ["data", "label"], shuffle=False)
#
# for data in dataset.create_dict_iterator():
#     print('{}'.format(data["data"]), '{}'.format(data["label"]))



class LeNet5(ms.nn.Cell):
    """
    Lenet网络结构
    """
    def __init__(self, num_class=10, num_channel=1):
        super(LeNet5, self).__init__()
        # 定义所需要的运算
        self.conv1 = ms.nn.Conv2d(num_channel, 6, 5, pad_mode='valid')
        self.conv2 = ms.nn.Conv2d(6, 16, 5, pad_mode='valid')
        self.fc1 = ms.nn.Dense(16 * 5 * 5, 120, weight_init=Normal(0.02))
        self.fc2 = ms.nn.Dense(120, 84, weight_init=Normal(0.02))
        self.fc3 = ms.nn.Dense(84, num_class, weight_init=Normal(0.02))
        self.relu = ms.nn.ReLU()
        self.max_pool2d = ms.nn.MaxPool2d(kernel_size=2, stride=2)
        self.flatten = ms.nn.Flatten()

    def construct(self, x):
        # 使用定义好的运算构建前向网络
        x = self.conv1(x)
        x = self.relu(x)
        x = self.max_pool2d(x)
        x = self.conv2(x)
        x = self.relu(x)
        x = self.max_pool2d(x)
        x = self.flatten(x)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.relu(x)
        x = self.fc3(x)
        return x



class DatasetGenerator:
    def __init__(self):
        self.data = np.random.sample((5, 2))
        self.label = np.random.sample((5, 1))

    def __getitem__(self, index):
        return self.data[index], self.label[index]

    def __len__(self):
        return len(self.data)

# def train_net(args, model, epoch_size, data_path, repeat_size, ckpoint_cb, sink_mode):
#     """定义训练的方法"""
#     # 加载训练数据集
#     ds_train = create_dataset(os.path.join(data_path, "train"), 32, repeat_size)
#     model.train(epoch_size, ds_train, callbacks=[ckpoint_cb, LossMonitor(125)], dataset_sink_mode=sink_mode)
#
# def test_net(network, model, data_path):
#     """定义验证的方法"""
#     ds_eval = create_dataset(os.path.join(data_path, "test"))
#     acc = model.eval(ds_eval, dataset_sink_mode=False)
#     print("{}".format(acc))



def main():

    parser = argparse.ArgumentParser(description='MindSpore LeNet Example')
    parser.add_argument('--device_target', type=str, default="CPU", choices=['Ascend', 'GPU', 'CPU'])

    args = parser.parse_known_args()[0]
    ms.context.set_context(mode=ms.context.GRAPH_MODE, device_target=args.device_target)



    dataset = init_dataset("Market-1501", root="/home/chenyifan/repos/Mirkwood/datasets")

    num_classes = dataset.num_train_pids
    train_set = ImageDataset(dataset)







    # 实例化网络
    net = LeNet5()

    # 定义损失函数
    net_loss = ms.nn.SoftmaxCrossEntropyWithLogits(sparse=True, reduction='mean')

    # 定义优化器
    net_opt = ms.nn.Momentum(net.trainable_params(), learning_rate=0.01, momentum=0.9)

    from mindspore.train.callback import ModelCheckpoint, CheckpointConfig
    # 设置模型保存参数
    config_ck = CheckpointConfig(save_checkpoint_steps=1875, keep_checkpoint_max=10)
    # 应用模型保存参数
    ckpoint = ModelCheckpoint(prefix="checkpoint_lenet", config=config_ck)

    train_epoch = 5
    mnist_path = "/home/chenyifan/datasets/mnist2"
    dataset_size = 1
    model = ms.Model(net, net_loss, net_opt, metrics={"Accuracy": Accuracy()})
    # train_net(args, model, train_epoch, mnist_path, dataset_size, ckpoint, False)
    # test_net(net, model, mnist_path)

    return 0


if __name__=="__main__":
    main()
