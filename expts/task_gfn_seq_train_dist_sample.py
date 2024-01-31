import sys
import itertools

root = "/mnt/ps/home/CORP/lazar.atanackovic/project/gflownet-runs/logs/gfn_seq_offline_09_train_dist_sample_Jan_15"
counter = itertools.count()

base_hps = {
    "num_training_steps": 100000,
    "validate_every": 200, # use 1000 might be faster
    "num_workers": 8,
    "pickle_mp_messages": True, # when using 1 or mor worker always have this True (otherwise slow)
    "model": {
        "num_layers": 8, 
        "num_emb": 128,
        "graph_transformer": {
            "num_heads": 4,
            },
        },
    "device": 'cuda',
}


base_algo_hps = {
    "global_batch_size": 256, # 256
    "max_nodes": 15,
    "max_len": 16, # needs to be max_len = max_nodes + 1
    "offline_ratio": 1, # 1
}

hps = [
    {
        **base_hps,
        "log_dir": f"{root}/run_{next(counter)}/",
        "log_tags": ["gfn_seq_offline_09_train_dist_sample"],
        "seed": seed,
        "opt": {"learning_rate": 1e-4}, # o.g. 1e-4,
        
        "task": {
        "toy_seq": {
            "test_split_seed": seed, 
            "do_supervised": False, 
            "do_tabular_model": False, 
            "regress_to_P_F": False,
            "regress_to_Fsa": False,
            "train_ratio": 0.9, # set this to 1 maybe??
            "reward_func": 'edit', 
            },
        },  
        
        "algo": {
            **base_algo_hps,
            "offline_sampling_g_distribution": data_distribution,
            "use_true_log_Z": False,
            **algo,
        },
        
    }
    for data_distribution in ["uniform", "log_rewards", "log_p", "l2_log_error_gfn", "l1_error_gfn"] # TODO implement "loss_gfn"
    for seed in [1, 2, 3]
    for algo in [
        {
            "method": "TB", # either TB or FM
            "tb": {
                "variant": "SubTB1", #"TB", 
                "do_parameterize_p_b": False,
                "cum_subtb": False,
                },
        },
    ]
]

from gflownet.tasks.toy_seq import ToySeqTrainer

trial = ToySeqTrainer(hps[int(sys.argv[1])])
trial.print_every = 1
trial.run()