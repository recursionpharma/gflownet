import sys
import itertools

#root = "/mnt/ps/home/CORP/lazar.atanackovic/project/gflownet-runs/logs/gfn_mixed_sample_count_Dec_1"
#root = "/mnt/ps/home/CORP/lazar.atanackovic/project/gflownet-runs/logs/gfn_mixed_sample_cliques_Jan_12"
root = "/mnt/ps/home/CORP/lazar.atanackovic/project/gflownet-runs/logs/gfn_mixed_sample_neighbors_Jan_23"
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
    #"opt": {"learning_rate": 1e-4}, # o.g. 1e-4
    "device": 'cuda',
}


base_algo_hps = {
    "global_batch_size": 256,
    "max_nodes": 7,
    "offline_ratio": 0,
}

hps = [
    {
        **base_hps,
        "log_dir": f"{root}/run_{next(counter)}/",
        "log_tags": ["gfn_mixed_sample_v3"],
        "seed": seed,
        "opt": {"learning_rate": 1e-4}, # o.g. 1e-4,
        
        "task": {
        "basic_graph": {
            "test_split_seed": 1, #142857, 
            "do_supervised": False, 
            "do_tabular_model": False, 
            "regress_to_P_F": False,
            "regress_to_Fsa": False,
            "reward_reshape": True,
            "train_ratio": 0.9, # set this to 1 maybe??
            "reward_func": reward, 
            },
        },  
        
        "algo": {
            **base_algo_hps,
            "dir_model_pretrain_for_sampling": "/mnt/ps/home/CORP/lazar.atanackovic/project/gflownet-runs/logs/gfn_SubTB_flows_Nov_21/run_0/model_state.pt",
            "alpha": alpha,
            **algo,
        },
        
    }
    for reward in ['even_neighbors'] #['count', 'even_neighbors', 'cliques']
    for seed in [1]
    for alpha in [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    #for lr in [5e-4, 1e-4, 5e-5, 1e-5]
    for algo in [
        {
            "method": "TB", # either TB or FM
            "tb": {
                "variant": "SubTB1", #"SubTB1", 
                "do_parameterize_p_b": False,
                },
        },
    ]
]

from gflownet.tasks.basic_graph_task import BasicGraphTaskTrainer

trial = BasicGraphTaskTrainer(hps[int(sys.argv[1])])
trial.print_every = 1
trial.run()