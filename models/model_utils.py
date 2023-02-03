from mindspore import nn
from mindspore.nn.learning_rate_schedule import LearningRateSchedule


class DBNetWithLossCell(nn.Cell):
    def __init__(self, backbone, loss_fn):
        super().__init__()
        self._backbone = backbone
        self._loss_fn = loss_fn

    def construct(self, data, *labels):
        output = self._backbone(data)
        return self._loss_fn(output, labels)


class DBNetWithEvalCell(nn.Cell):
    def __init__(self, network):
        super().__init__()
        self._network = network

    def construct(self, data, label1, label2):
        output = self._network(data)
        # TODO: perform boxes and scores extraction here?
        return output, label1, label2


class DynamicLR(LearningRateSchedule):
    def __init__(self, learning_rate, warmup_epochs, end_learning_rate, decay_epochs, power, steps_per_epoch):
        super().__init__()
        self._warmup_steps = warmup_epochs * steps_per_epoch
        if warmup_epochs:
            self._warmup_lr = nn.WarmUpLR(learning_rate, self._warmup_steps)
        decay_steps = decay_epochs * steps_per_epoch
        self._poly_lr = nn.PolynomialDecayLR(learning_rate, end_learning_rate, decay_steps - self._warmup_steps, power)

    def construct(self, global_step):
        if global_step < self._warmup_steps:
            return self._warmup_lr(global_step)
        else:
            return self._poly_lr(global_step - self._warmup_steps)
