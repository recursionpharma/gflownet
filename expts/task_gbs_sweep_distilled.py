import sys
import itertools

root = "./logs/global_bs_distilled/"
counter = itertools.count()

base_hps = {
    "num_training_steps": 20000,
    "validate_every": 100,
    "num_workers": 4,
    "pickle_mp_messages": True, # when using 1 or mor worker always have this True (otherwise slow)
    "model": {"num_layers": 2, "num_emb": 256},
    "opt": {"adam_eps": 1e-8, "learning_rate": 3e-4},
    "device": 'cuda',
}

base_algo_hps = {
    "max_nodes": 7,
    "offline_ratio": 0 / 4,
}

hps = [
    {
        **base_hps,
        "log_dir": f"{root}/run_{next(counter)}/",
        "log_tags": ["global_bs_sweep"],
        
        "task": {
        "basic_graph": {
            "test_split_seed": seed, 
            "do_supervised": True, 
            "do_tabular_model": False, 
            "regress_to_P_F": True,
            "regress_to_Fsa": True,
            "train_ratio": 0.9,
            "reward_func": reward, 
            },
        },  
        
        "algo": {
            **base_algo_hps,
            #**algo,
            "global_batch_size": global_bs,
        },
        
    }
    for reward in ['const', 'count', 'even_neighbors', 'cliques']
    for seed in [1, 2, 3, 4, 5]
    for global_bs in [16, 32, 64, 128, 256]
    #for algo in [
        #{
        #    "method": "TB", # either TB or FM
        #    "tb": {"variant": "SubTB1", "do_parameterize_p_b": False},
        #},
        #{
        #    "method": "FM", # either TB or FM
        #},
    #]
]

from gflownet.tasks.basic_graph_task import BGSupervisedTrainer

trial = BGSupervisedTrainer(hps[int(sys.argv[1])])
trial.print_every = 1
trial.run()