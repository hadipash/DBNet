import mindspore as ms
from mindspore import nn

from models import create_model
from losses.total_dbnet_loss import L1BalanceCELoss
from models.model_utils import CustomWithLossCell, CustomWithEvalCell, DynamicLR
from datasets.load import create_dataset
from metrics.metric import MyMetric
from callbacks import ValCallback

ms.set_context(mode=ms.GRAPH_MODE, device_target="Ascend")

EPOCHS = 1000

net = create_model('dbnet', train=True)

dataset, _ = create_dataset(True)
val_dataset, _ = create_dataset(False)

lr = DynamicLR(learning_rate=0.007, warmup_epochs=5, end_learning_rate=0., decay_epochs=EPOCHS, power=0.9,
               steps_per_epoch=dataset.get_dataset_size())
opt = nn.Momentum(net.trainable_params(), learning_rate=lr, momentum=0.9, weight_decay=1e-4)

model = ms.Model(network=CustomWithLossCell(net, L1BalanceCELoss()), optimizer=opt,
                 metrics={'Test': MyMetric()}, eval_network=CustomWithEvalCell(net))

# profiler = ms.Profiler(output_path='./profiler/eval1_dcn')
model.train(EPOCHS, dataset,
            callbacks=[ms.LossMonitor(dataset.get_dataset_size()), ms.TimeMonitor(), ValCallback(model, val_dataset),
                       ms.ModelCheckpoint(directory='ckpts', config=ms.CheckpointConfig(async_save=True))])
# profiler.analyse()
