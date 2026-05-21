from typing import Any


class IPCModule:
    """Inter-Process Communication Module between GFN sampler and Reward Module"""

    def __init__(self, process_type: str):
        """Initialization of IPC Module

        Parameters
        ----------
        workdir : str | Path
            working directory of IPCModule
        process_type : str
            - GFNTask: 'sampler'
            - RewardModule: 'reward'

        """
        self.process_type: str = process_type
        assert self.process_type in (
            "sampler",
            "reward",
        ), f"process ({self.process_type}) should be 'sampler' or 'reward"

    @property
    def is_sampler(self):
        return self.process_type == "sampler"

    @property
    def is_reward(self):
        return self.process_type == "reward"

    def assert_is_sampler(self):
        assert self.is_sampler

    def assert_is_reward(self):
        assert self.is_reward

    ### GFN ###
    def sampler_from_reward(self) -> tuple[list[list[float]], list[bool]]:
        """Response from Reward Module for n sampled objects

        Returns
        -------
        Tuple[list[list[float]], list[bool]
            - obj_props: [m, n_objectives]
                obj_props of n<=m valid objects
            - is_valid: [n,]
                flags whether the objects are valid or not
        """
        raise NotImplementedError

    def sampler_to_reward(self, objs: list[Any]) -> bool:
        """Request to Reward Module for rewarding

        Parameters
        ----------
        objs : list[Any]
            A List of n sampled objects

        Returns
        -------
        status: bool
        """
        raise NotImplementedError

    def sampler_wait_reward(self) -> bool:
        """Wait to receive signal from Reward."""
        raise NotImplementedError

    def sampler_unlock_reward(self):
        """Send signal to Reward."""
        raise NotImplementedError

    def sampler_terminate_reward(self):
        """Send signal to Reward to be terminate"""
        raise NotImplementedError

    ### Reward Module ###
    def reward_from_sampler(self) -> list[Any]:
        """Receive objects from Sampler process
        if the gflownet process is termiated, it will return None"""
        raise NotImplementedError

    def reward_to_sampler(self, obj_props: list[list[float]], is_valid: list[bool]) -> None:
        """Send rewards to Sampler process

        Parameters
        ----------
        obj_props: list[list[float]], [m, n_objectives]
            obj_props of n<=m valid objects
        is_valid: list[bool], [n,]
            flags whether the objects are valid or not
        """
        raise NotImplementedError

    def reward_wait_sampler(self) -> bool:
        """Wait to receive signal from Sampler.
        if it return False, the rewarding is finished
        """
        raise NotImplementedError

    def reward_unlock_sampler(self):
        """Send signal to Sampler."""
        raise NotImplementedError

    def reward_is_terminated(self):
        """Check the sampler is termiated or not"""
        raise NotImplementedError
