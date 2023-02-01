from mindspore.train.callback import Callback


class ValCallback(Callback):
    def __init__(self, model, val_dataset):
        self._model = model
        self._data = val_dataset

    def on_train_epoch_end(self, run_context):
        cb_params = run_context.original_args()
        epoch_num = cb_params.cur_epoch_num
        if epoch_num > 0 and epoch_num % 5 == 0:
            metrics = self._model.eval(self._data)['Test']
            print(f"Recall: {metrics['recall'].avg} "
                  f"Precision: {metrics['precision'].avg} "
                  f"Fmeasure: {metrics['fmeasure'].avg}")
