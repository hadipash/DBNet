import mindspore as ms
from mindspore import nn

from models import create_model
from losses.total_dbnet_loss import L1BalanceCELoss
from models.model_utils import CustomWithLossCell, CustomWithEvalCell
from datasets.load import create_dataset
from metrics.metric import MyMetric

ms.set_context(mode=ms.PYNATIVE_MODE, device_target="Ascend")

net = create_model('dbnet', train=True)
opt = nn.Momentum(net.trainable_params(), learning_rate=0.005, momentum=0.9)

dataset, _ = create_dataset(True)
val_dataset, _ = create_dataset(False)

model = ms.Model(network=CustomWithLossCell(net, L1BalanceCELoss()), optimizer=opt,
                 metrics={'Test': MyMetric()}, eval_network=CustomWithEvalCell(net))

# model.train(1, dataset, callbacks=[ms.LossMonitor(20)])
metrics = model.eval(val_dataset)['Test']
print(f"Recall: {metrics['recall'].avg}\n"
      f"Precision: {metrics['precision'].avg}\n"
      f"Fmeasure: {metrics['fmeasure'].avg}\n")
