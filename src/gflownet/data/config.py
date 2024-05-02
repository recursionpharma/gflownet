from dataclasses import dataclass
from typing import Optional

from gflownet.utils.misc import StrictDataClass


@dataclass
class ReplayConfig(StrictDataClass):
    """Replay buffer configuration

    Attributes
    ----------
    use : bool
        Whether to use a replay buffer
    capacity : int
        The capacity of the replay buffer
    warmup : int
        The number of samples to collect before starting to sample from the replay buffer
    hindsight_ratio : float
        The ratio of hindsight samples within a batch
    num_from_replay : Optional[int]
        The number of replayed samples for a training batch (defaults to cfg.algo.num_from_policy, i.e. a 50/50 split)
    num_new_samples : Optional[int]
        The number of new samples added to the replay at every training step. Defaults to cfg.algo.num_from_policy. If
        smaller than num_from_policy then not all on-policy samples will be added to the replay. If larger
        than num_from_policy then the training batch will not contain all the new samples, but the buffer will.
        For example, if one wishes to sample N samples every step but only add them to the buffer and not make them
        part of the training batch, then one should set replay.num_new_samples=N and algo.num_from_policy=0.
    """

    use: bool = False
    capacity: Optional[int] = None
    warmup: Optional[int] = None
    hindsight_ratio: float = 0
    num_from_replay: Optional[int] = None
    num_new_samples: Optional[int] = None
