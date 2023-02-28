import mindspore as ms
from mindcv import create_scheduler, create_optimizer

from models import create_model
from losses.total_dbnet_loss import L1BalanceCELoss
from models.model_utils import DBNetWithLossCell, DBNetWithEvalCell
from datasets import create_dataset
from metrics import DBNetMetric
from callbacks import EpochSummary


def train(config):
    ms.set_context(mode=ms.PYNATIVE_MODE if config['env']['mode'].lower() == 'pynative' else ms.GRAPH_MODE,
                   device_target=config['env']['device_target'], device_id=config['env']['device_id'])

    net = create_model(config['model'])

    train_dataset = create_dataset(config['dataset'], stage='train')
    val_dataset = create_dataset(config['dataset'], stage='val') if 'val' in config['dataset'] else None

    lr = create_scheduler(train_dataset.get_dataset_size(), scheduler=config['train']['scheduler']['name'],
                          num_epochs=config['train']['epochs'], **config['train']['scheduler']['params'])

    opt = create_optimizer(net.trainable_params(), opt=config['train']['optimizer']['name'], lr=lr,
                           **config['train']['optimizer']['params'])

    model = ms.Model(network=DBNetWithLossCell(net, L1BalanceCELoss()), optimizer=opt,
                     metrics={'Eval': DBNetMetric()}, eval_network=DBNetWithEvalCell(net))

    profiler = ms.Profiler(output_path=config['env']['profiler']) if config['env']['profiler'] else None
    model.train(config['train']['epochs'], train_dataset, dataset_sink_mode=False,
                callbacks=[EpochSummary(model, val_dataset, val_freq=config.get('val', {}).get('eval_freq'),
                                        folder=config['model']['save_ckpt'])])
    if profiler is not None:
        profiler.analyse()


if __name__ == '__main__':
    import yaml
    import argparse
    from shutil import copy
    from pathlib import Path

    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config', help='path to the configuration yaml file')
    args = parser.parse_args()

    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)

    folder = Path(config['model']['save_ckpt'])
    folder.mkdir(parents=True, exist_ok=True)
    print(f'Saving config and weights to {str(folder.resolve())}')
    copy(args.config, folder)

    train(config)
