import mindspore
from mindspore import context
from mindspore import Tensor
from mindspore import set_seed
from mindspore import dtype as mstype
from mindspore.common.initializer import One, Normal
from src.resnet import resnet50
from mindspore import nn
from mindspore import load_checkpoint, load_param_into_net

def main():
    context.set_context(mode=context.GRAPH_MODE,
                        device_target="CPU",
                        device_id=0)
    resnet = resnet50(1001)

    model_path = "/home/chenyifan/repos/Mirkwood/DAAF-trinet-mindspore/pretrain-model/resnet50_ascend_v120_imagenet2012_official_cv_bs256_acc76.ckpt"
    # initialize the number of classes based on the pre-trained model

    param_dict = load_checkpoint(model_path)

    load_param_into_net(resnet, param_dict)

    tensor2 = Tensor(shape=(100, 100), dtype=mstype.float32, init=Normal())



if __name__ == '__main__':
    rank = 0
    device_num = 1

    # 调用接口进行数据处理
    dataset = create_new_dataset(image_dir=config.coco_root, batch_size=config.batch_size, is_training=True, num_parallel_workers=8)
    dataset_size = dataset.get_dataset_size()
    print("total images num: ", dataset_size)
    print("Create dataset done!")

    # 实例化网络
    net = Mask_Rcnn_Resnet50(config=config)
    net = net.set_train()

    # 加载预训练模型
    load_path = args_opt.pre_trained
    if load_path != "":
        param_dict = load_checkpoint(load_path)
        if config.pretrain_epoch_size == 0:
            for item in list(param_dict.keys()):
                if not (item.startswith('backbone') or item.startswith('rcnn_mask')):
                    param_dict.pop(item)
        load_param_into_net(net, param_dict)

    # 设定损失函数、学习率、优化器
    loss = LossNet()
    opt = nn.Momentum(params=net.trainable_params(), learning_rate=0.0001, momentum=config.momentum,
                   weight_decay=config.weight_decay, loss_scale=config.loss_scale)

    # 包装损失函数
    net_with_loss = WithLossCell(net, loss)

    # 通过TrainOneStepCell自定义训练过程
    net = TrainOneStepCell(net_with_loss, opt, sens=config.loss_scale)

    # 监控训练过程
    time_cb = TimeMonitor(data_size=dataset_size)
    loss_cb = LossCallBack(rank_id=rank)
    cb = [time_cb, loss_cb]

    # 保存训练后的模型
    if config.save_checkpoint:
        # 设置模型保存参数
        ckptconfig = CheckpointConfig(save_checkpoint_steps=config.save_checkpoint_epochs * dataset_size,
                                      keep_checkpoint_max=config.keep_checkpoint_max)
        save_checkpoint_path = os.path.join(config.save_checkpoint_path, 'ckpt_' + str(rank) + '/')
        # 应用模型保存参数
        ckpoint_cb = ModelCheckpoint(prefix='mask_rcnn', directory=save_checkpoint_path, config=ckptconfig)
        cb += [ckpoint_cb]

    # 进行训练
    model = Model(net)
    model.train(config.epoch_size, dataset, callbacks=cb, dataset_sink_mode = False)