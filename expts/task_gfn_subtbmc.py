import sys
import itertools

root = "/mnt/ps/home/CORP/lazar.atanackovic/project/gflownet-runs/logs/gfn_SubTBMC_high_lr_Jan_10"
counter = itertools.count()

base_hps = {
    "num_training_steps": 100000,
    "validate_every": 1000, # use 1000 might be faster
    "num_workers": 8,
    "pickle_mp_messages": True, # when using 1 or mor worker always have this True (otherwise slow)
    "model": {
        "num_layers": 8, 
        "num_emb": 128,
        "graph_transformer": {
            "num_heads": 4,
            "num_mlp_layers": 2, 
            },
        },
    "opt": {"learning_rate": 1e-3},
    "device": 'cuda',
}


base_algo_hps = {
    "global_batch_size": 256,
    "max_nodes": 7,
    "offline_ratio": 0 / 4,
}

hps = [
    {
        **base_hps,
        "log_dir": f"{root}/run_{next(counter)}/",
        "log_tags": ["gfn_subtbmc_high_lr"],
        "seed": seed,
        
        "task": {
        "basic_graph": {
            "test_split_seed": seed, 
            "do_supervised": False, 
            "do_tabular_model": False, 
            "regress_to_P_F": False,
            "regress_to_Fsa": False,
            "train_ratio": 0.9,
            "reward_func": reward, 
            },
        },  
        
        "algo": {
            **base_algo_hps,
            **algo,
        },
        
    }
    for reward in ['count', 'even_neighbors', 'cliques']
    for seed in [1, 2, 3]
    for algo in [
        {
            "method": "TB", # either TB or FM
            "tb": {
                "variant": "TB", 
                "do_parameterize_p_b": False
                },
        },
        {
            "method": "TB", # either TB or FM
            "tb": {
                "variant": "SubTB1", 
                "do_parameterize_p_b": False,
                "cum_subtb": False,
                },
        },
        {
            "method": "TB", # either TB or FM
            "tb": {
                "variant": "SubTBMC", 
                "do_parameterize_p_b": False, 
                "cum_subtb": False ,
                },
        },
    ]
]

from gflownet.tasks.basic_graph_task import BasicGraphTaskTrainer

trial = BasicGraphTaskTrainer(hps[int(sys.argv[1])])
trial.print_every = 1
trial.run()