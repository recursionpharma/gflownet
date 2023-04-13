from itertools import count
import os
import sys

import torch

from gflownet.tasks.seh_frag import SEHFragTrainer

log_root = '/rxrx/data/user/emmanuel.bengio/logs'

# One such run takes about 4 hours on our cluster
base_hps = {
    'num_training_steps': 20_000,
    'global_batch_size': 64,
    'validate_every': 500,
    'lr_decay': 20000,
    'num_data_loader_workers': 8,
    'temperature_dist_params': 32.0,
    'temperature_sample_dist': 'constant',
    'sampling_tau': 0.99,
    'mp_pickle_messages': True,
}

counter = count(0)
apr_5 = [
    {
        **base_hps,
        'learning_rate': lr,
        'log_dir': f'{log_root}/seh_frag/apr_5_test_{next(counter)}/',
    }  #
    for seed in range(4)  #
    for lr in [3e-4, 1e-4]  #
]

if __name__ == '__main__':
    array = eval(sys.argv[1])
    hps = array[int(sys.argv[2])]
    os.makedirs(hps['log_dir'], exist_ok=True)
    trial = SEHFragTrainer(hps, torch.device('cuda'))
    trial.verbose = True
    trial.run()
