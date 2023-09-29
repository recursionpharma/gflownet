import sys

root = "/mnt/ps/home/CORP/lazar.atanackovic/project/gflownet-runs/logs/gfn_single_runs"

base_hps = {
    "num_training_steps": 50000,
    "validate_every": 100,
    "num_workers": 4,
    "pickle_mp_messages": True, # when using 1 or mor worker always have this True (otherwise slow)
    "model": {
        "num_layers": 8, 
        "num_emb": 128,
        "graph_transformer": {
            "num_heads": 4,
            "num_mlp_layers": 2, 
            },
        },
    "opt": {"learning_rate": 1e-4},
    "device": 'cuda',
}

base_algo_hps = {
    "global_batch_size": 64,
    "max_nodes": 7,
    "offline_ratio": 0 / 4,
}

hps = [
    {
        **base_hps,
        "log_dir": f"{root}/run_gfn/",
        "log_tags": ["test_tag_cfg"],
        
        "task": {
        "basic_graph": {
            "test_split_seed": 0, 
            "do_supervised": True, 
            "do_tabular_model": False, 
            "regress_to_P_F": True,
            "regress_to_Fsa": True,
            "train_ratio": 0.9,
            "reward_func": 'count', 
            "reward_reshape": False,
            "reward_corrupt": False,
            "reward_shuffle": False,
            "reward_temper": True,
            "reward_param": 0.1,
            },
        },  
        
        #"algo": {
        #    **base_algo_hps,
        #    "method": "TB", # either TB or FM
        #    "tb": {"variant": "SubTB1", "do_parameterize_p_b": False},
        #},
        
    }
]

from gflownet.tasks.basic_graph_task import BGSupervisedTrainer

trial = BGSupervisedTrainer(hps[0])
trial.print_every = 1
trial.run()