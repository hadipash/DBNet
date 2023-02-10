import mindspore as ms
from mindspore import nn

from models import create_model
from losses.total_dbnet_loss import L1BalanceCELoss
from models.model_utils import DBNetWithLossCell, DBNetWithEvalCell, DynamicLR
from datasets import create_dataset
from metrics.metric import DBNetMetric
from callbacks import EpochSummary

ms.set_context(mode=ms.GRAPH_MODE, device_target="Ascend")

EPOCHS = 1200

net = create_model('dbnet', train=True)

dataset, _ = create_dataset(True)
val_dataset, _ = create_dataset(False)

lr = DynamicLR(learning_rate=0.007, warmup_epochs=3, end_learning_rate=0., decay_epochs=EPOCHS, power=0.9,
               steps_per_epoch=dataset.get_dataset_size())
opt = nn.SGD(net.trainable_params(), learning_rate=lr, momentum=0.9, weight_decay=1e-4)

model = ms.Model(network=DBNetWithLossCell(net, L1BalanceCELoss()), optimizer=opt,
                 metrics={'Eval': DBNetMetric()}, eval_network=DBNetWithEvalCell(net))

# profiler = ms.Profiler(output_path='./profiler/eval1_dcn')
model.train(EPOCHS, dataset, callbacks=[EpochSummary(model, val_dataset, folder='./ckpts/')],
            dataset_sink_mode=False)
# profiler.analyse()
