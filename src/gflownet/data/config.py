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
    """

    use: bool = False
    capacity: Optional[int] = None
    warmup: Optional[int] = None
    hindsight_ratio: float = 0
