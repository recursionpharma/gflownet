from dataclasses import dataclass
from typing import Optional


@dataclass
class ReplayConfig:
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
    batch_size : Optional[int]
        The batch size for sampling from the replay buffer, defaults to the online batch size
    replaces_online_data : bool
        Whether to replace online data with samples from the replay buffer
    """

    use: bool = False
    capacity: Optional[int] = None
    warmup: Optional[int] = None
    hindsight_ratio: float = 0
    batch_size: Optional[int] = None
    replaces_online_data: bool = True
