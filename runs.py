import os
import time

os.environ["DEVICE_ID"] = "1"

import mindspore as ms

ms.set_context(mode=ms.PYNATIVE_MODE)


def run_backbone():
    from backbones.resnet_dcn import DBNetResNet, DBNetResNetDCN, Bottleneck

    xin = ms.ops.ones((4, 3, 224, 224), type=ms.float32)
    backbone = DBNetResNetDCN(Bottleneck, [3, 4, 6, 3])
    # backbone = mindcv.create_model('resnet50')
    for feature in backbone(xin):
        print(feature.shape)


def run_dbnet():
    from backbones.resnet_dcn import DBNetResNet, DBNetResNetDCN, Bottleneck
    from models.dbnet import DBNet, DBNetPP

    # ms.set_seed(42)

    xin = ms.ops.ones((1, 3, 736, 1280), type=ms.float32)

    net = DBNetPP(backbone=DBNetResNetDCN(Bottleneck, [3, 4, 6, 3]), in_channels=[256, 512, 1024, 2048], adaptive=False)
    # net = net.backbone
    net.set_train(False)
    # print(net)
    # yout = net(xin)

    # warmup
    for _ in range(10):
        yout = net(xin)
    print('Finished warmup')

    shapes = []
    times = []
    profiler = ms.Profiler(output_path='./profiler/rename')
    for _ in range(100):
        start = time.perf_counter()
        yout = net(xin)[0]
        shapes.append(yout.shape)
        times.append(time.perf_counter() - start)
    profiler.analyse()
    print(f'fps: {len(times) / sum(times)}')
    print(f'Average time: {sum(times) / len(times)}')


def run_dataset():
    from datasets.load import create_dataset

    train_dataset, steps_pre_epoch = create_dataset(True)
    gg = next(train_dataset.create_tuple_iterator())
    zz = 0


def run_loss():
    import numpy as np
    from losses.balanced_bce_loss import BalancedCrossEntropyLoss


    # pred = ms.ops.uniform((2, 1, 640, 640), minval=ms.Tensor(0.), maxval=ms.Tensor(2.), seed=42)
    pred = ms.Tensor.from_numpy(np.random.uniform(size=(2, 1, 640, 640)).astype(np.float32))
    gt = ms.Tensor.from_numpy(np.random.randint(0, 2, size=(2, 1, 640, 640)).astype(np.float32))
    mask = ms.Tensor.from_numpy(np.random.randint(0, 2, size=(2, 640, 640)).astype(np.float32))

    loss = BalancedCrossEntropyLoss()
    loss_out = loss(pred, gt, mask)


if __name__ == '__main__':
    run_dataset()
