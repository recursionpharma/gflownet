from pathlib import Path

import torch
import torch.nn as nn

from gflownet.utils.metrics import get_limits_of_hypercube


class FocusModel:
    """
    Abstract class for a belief model over focus directions for goal-conditioned GFNs.
    Goal-conditioned GFNs allow for more control over the objective-space region from which
        we wish to sample. However due to the growing number of emtpy regions in the objective space,
        if we naively sample focus-directions from the entire objective space, we will condition
        our GFN with a lot of infeasible directions which significantly harms its sample efficiency
        compared to a more simple preference-conditioned model.
    To alleviate this problem, we introduce a focus belief model which is used to sample
        focus directions from a subset of the objective space. The belief model is
        trained to predict the probability of a focus direction being feasible. The likelihood
        to sample a focus direction is then proportional to its population. Directions that have never
        been sampled should be given the maximum likelihood.
    """

    def __init__(self, device: torch.device, n_objectives: int, state_space_res: int) -> None:
        """
        args:
            device: torch device
            n_objectives: number of objectives
            state_space_res: resolution of the state space discretisation. The number of focus directions to consider
                grows within O(state_space_res ** n_objectives) and depends on the amount of filtering we apply
                (e.g. valid focus-directions should sum to 1 [dirichlet], should contain a 1 [limits], etc.)
        """
        self.device = device
        self.n_objectives = n_objectives
        self.state_space_res = state_space_res

        self.feasible_flow = 1.0
        self.infeasible_flow = 0.1

    def update_belief(self, focus_dirs: torch.Tensor, flat_rewards: torch.Tensor):
        raise NotImplementedError

    def sample_focus_directions(self, n: int):
        raise NotImplementedError


class TabularFocusModel(FocusModel):
    """
    Tabular model of the feasibility of focus directions for goal-condtioning.
    We keep a count of the number of times each focus direction has been sampled and whether
    this direction succesfully lead to a sample in this region of the objective space. The (unormalized) likelihood
    of a focus direction being feasible is then given by the ratio of these numbers.
    If a focus direction has not been sampled yet it obtains the maximum likelihood of one.
    """

    def __init__(self, device: torch.device, n_objectives: int, state_space_res: int) -> None:
        super().__init__(device, n_objectives, state_space_res)
        self.n_objectives = n_objectives
        self.state_space_res = state_space_res
        self.focus_dir_dataset = (
            nn.functional.normalize(torch.tensor(get_limits_of_hypercube(n_objectives, state_space_res)), dim=1)
            .float()
            .to(self.device)
        )
        self.focus_dir_count = torch.zeros(self.focus_dir_dataset.shape[0]).to(self.device)
        self.focus_dir_population_count = torch.zeros(self.focus_dir_dataset.shape[0]).to(self.device)

    def update_belief(self, focus_dirs: torch.Tensor, flat_rewards: torch.Tensor):
        """
        Updates the focus model with the focus directions and rewards
        of the last batch.
        """
        focus_dirs = nn.functional.normalize(focus_dirs, dim=1)
        flat_rewards = nn.functional.normalize(flat_rewards, dim=1)

        focus_dirs_indices = torch.argmin(torch.cdist(focus_dirs, self.focus_dir_dataset), dim=1)
        flat_rewards_indices = torch.argmin(torch.cdist(flat_rewards, self.focus_dir_dataset), dim=1)

        for idxs, count in zip(
            [focus_dirs_indices, flat_rewards_indices],
            [self.focus_dir_count, self.focus_dir_population_count],
        ):
            idx_increments = torch.bincount(idxs, minlength=len(count))
            count += idx_increments

    def sample_focus_directions(self, n: int):
        """
        Samples n focus directions from the focus model.
        """
        sampling_likelihoods = torch.zeros_like(self.focus_dir_count).float().to(self.device)
        sampling_likelihoods[self.focus_dir_count == 0] = self.feasible_flow
        sampling_likelihoods[torch.logical_and(self.focus_dir_count > 0, self.focus_dir_population_count > 0)] = (
            self.feasible_flow
        )
        sampling_likelihoods[torch.logical_and(self.focus_dir_count > 0, self.focus_dir_population_count == 0)] = (
            self.infeasible_flow
        )
        focus_dir_indices = torch.multinomial(sampling_likelihoods, n, replacement=True)
        return self.focus_dir_dataset[focus_dir_indices].to("cpu")

    def save(self, path: Path):
        params = {
            "n_objectives": self.n_objectives,
            "state_space_res": self.state_space_res,
            "focus_dir_dataset": self.focus_dir_dataset.to("cpu"),
            "focus_dir_count": self.focus_dir_count.to("cpu"),
            "focus_dir_population_count": self.focus_dir_population_count.to("cpu"),
        }
        torch.save(params, open(path / "tabular_focus_model.pt", "wb"))

    def load(self, device: torch.device, path: Path):
        params = torch.load(open(path / "tabular_focus_model.pt", "rb"))
        self.n_objectives = params["n_objectives"]
        self.state_space_res = params["state_space_res"]
        self.focus_dir_dataset = params["focus_dir_dataset"].to(device)
        self.focus_dir_count = params["focus_dir_count"].to(device)
        self.focus_dir_population_count = params["focus_dir_population_count"].to(device)
