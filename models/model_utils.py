from mindspore import nn


class CustomWithLossCell(nn.Cell):
    def __init__(self, backbone, loss_fn):
        super().__init__()
        self._backbone = backbone
        self._loss_fn = loss_fn

    def construct(self, data, *labels):
        output = self._backbone(data)
        return self._loss_fn(output, labels)


class CustomWithEvalCell(nn.Cell):
    def __init__(self, network):
        super().__init__()
        self._network = network

    def construct(self, data, *labels):
        output = self._network(data)
        return output, labels
