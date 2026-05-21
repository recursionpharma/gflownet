from dataclasses import dataclass, field

from omegaconf import MISSING


@dataclass
class NetworkIPCConfig:
    """See communicate.method.network.NetworkIPC

    Attributes
    ----------
    host: str
        Host for networking. should be equal for gfn and reward processes
    port: int
        Port for networking. should be equal for gfn and reward processes
    """

    host: str = "localhost"
    port: int = MISSING


@dataclass
class FileSystemIPCConfig:
    """See communicate.method.filesystem.FileSystemIPC

    Attributes
    ----------
    log_dir : str
        The directory where to communicate. should be equal for gfn and reward processes
    overwrite_existing_exp : bool
        Whether to overwrite the contents of the workdir if it already exists.
        This is unsafe
    num_objectives: int
        Only for FileSystemIPC_CSV.
    """

    workdir: str = MISSING
    overwrite_existing_exp: bool = False
    num_objectives: int = MISSING


@dataclass
class CommunicationConfig:
    """Configuration for Inter-Process Communication.

    Attributes
    ----------
    method : str
        IPC method.
        implementation list:
            'network': Using Network socket
            'file': Using FileSystem with serialized data
            'file-csv': Using FileSystem with csv-format text data

    """

    method: str = "network"
    network: NetworkIPCConfig = field(default_factory=NetworkIPCConfig)
    filesystem: FileSystemIPCConfig = field(default_factory=FileSystemIPCConfig)
