from pathlib import Path

import numpy as np
from tqdm import tqdm
from mindspore import save_checkpoint
from mindspore.train.callback import Callback


class EpochSummary(Callback):
    def __init__(self, model, val_dataset, val_freq=5, folder='./ckpts'):
        self._model = model
        self._data = val_dataset
        self._val_freq = val_freq
        self._folder = Path(folder)
        self._folder.mkdir(parents=True, exist_ok=True)
        self._max_f = 0.
        self._losses = []
        # a placeholder for the train progress meter. The actual initialization is in `on_train_epoch_begin`
        self._pbar = tqdm(disable=True)

    def on_train_step_end(self, run_context):
        self._losses.append(run_context.original_args().net_outputs.asnumpy())
        self._pbar.update()
        self._pbar.set_postfix_str(f'Loss: {self._losses[-1]:.4f}')

    def on_train_epoch_begin(self, run_context):
        self._pbar.close()
        self._pbar = tqdm(total=run_context.original_args().batch_num, ncols=140)
        self._pbar.set_description(f'Epoch {run_context.original_args().cur_epoch_num}')
        self._losses = []

    def on_train_epoch_end(self, run_context):
        cb_params = run_context.original_args()
        epoch = cb_params.cur_epoch_num

        loss = np.average(self._losses)
        if epoch % self._val_freq == 0:
            metrics = self._model.eval(self._data, dataset_sink_mode=False)['Eval']
            log = f"Loss: {loss:.4f} | Recall: {metrics['recall'].avg:.4f} | " \
                  f"Precision: {metrics['precision'].avg:.4f} | F-score: {metrics['fmeasure'].avg:.4f}"

            if metrics['fmeasure'].avg > self._max_f:
                self._max_f = metrics['fmeasure'].avg
                file_name = str(self._folder / f"epoch{epoch}_l{loss:.4f}_r{metrics['recall'].avg:.4f}_"
                                               f"p{metrics['precision'].avg:.4f}_f{metrics['fmeasure'].avg:.4f}.ckpt")
                # Save the network model.
                save_checkpoint(cb_params.train_network, ckpt_file_name=file_name, async_save=True)
        else:
            log = f"Loss: {loss:.4f}"

        self._pbar.set_postfix_str(log)

    def __exit__(self, *err):
        self._pbar.close()
