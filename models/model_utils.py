from mindspore import nn


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
