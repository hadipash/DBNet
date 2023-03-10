from pathlib import Path

import numpy as np
from tqdm import tqdm
from mindspore import save_checkpoint
from mindspore.train.callback import Callback


class EpochSummary(Callback):
    def __init__(self, model, val_dataset, val_freq=5, update_freq=1, folder='./ckpts'):
        self._model = model
        self._data = val_dataset
        self._val_freq = val_freq
        self._update_freq = update_freq
        self._folder = Path(folder)
        self._folder.mkdir(parents=True, exist_ok=True)
        self._max_f = 0.
        self._losses = []
        # a placeholder for the train progress meter. The actual initialization is in `on_train_epoch_begin`
        self._pbar = tqdm(disable=True)
        self._pbar_ncols = 100

    def on_train_step_end(self, run_context):
        cb_params = run_context.original_args()
        self._losses.append(cb_params.net_outputs.asnumpy())
        if cb_params.cur_step_num % self._update_freq == 0:
            log = f'Loss: {self._losses[-1]:.4f}'
            self._pbar.ncols = self._pbar_ncols + len(log)
            self._pbar.update(self._update_freq)
            self._pbar.set_postfix_str(log)

    def on_train_epoch_begin(self, run_context):
        self._pbar.close()
        self._pbar = tqdm(total=run_context.original_args().batch_num, ncols=self._pbar_ncols)
        self._pbar.set_description(f'Epoch {run_context.original_args().cur_epoch_num}')
        self._losses = []

    def on_train_epoch_end(self, run_context):
        cb_params = run_context.original_args()
        epoch = cb_params.cur_epoch_num

        loss = np.average(self._losses)
        if self._data is not None and epoch % self._val_freq == 0:
            metrics = self._model.eval(self._data, dataset_sink_mode=False)['Eval']
            log = f"Loss: {loss:.4f} | Recall: {metrics['recall']:.4f} | " \
                  f"Precision: {metrics['precision']:.4f} | F-score: {metrics['f-score']:.4f}"

            if metrics['f-score'] > self._max_f:
                self._max_f = metrics['f-score']
                file_name = str(self._folder / f"epoch{epoch}_l{loss:.4f}_r{metrics['recall']:.4f}_"
                                               f"p{metrics['precision']:.4f}_f{metrics['f-score']:.4f}.ckpt")
                # Save the network model.
                save_checkpoint(cb_params.train_network, ckpt_file_name=file_name, async_save=True)
        else:
            log = f"Loss: {loss:.4f}"

        self._pbar.ncols = self._pbar_ncols + len(log)
        self._pbar.set_postfix_str(log)

    def __exit__(self, *err):
        self._pbar.close()
