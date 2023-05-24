import torch
import torch.nn.functional
from torch_geometric.data import Batch, Data

from gflownet.envs.graph_building_env import GraphActionCategorical, GraphActionType


def make_test_cat():
    batch = Batch.from_data_list(
        [
            Data(x=torch.ones((2, 8)), y=torch.ones((1, 8))),
            Data(x=torch.ones((2, 8)), y=torch.ones((1, 8))),
            Data(x=torch.ones((2, 8)), y=torch.ones((0, 8))),
        ],
        follow_batch=["y"],
    )
    cat = GraphActionCategorical(
        # Let's use arange to have different logit values
        batch,
        logits=[
            torch.arange(3).reshape((3, 1)).float(),
            torch.arange(6 * 4).reshape((6, 4)).float(),
            torch.arange(2 * 3).reshape((2, 3)).float(),
        ],
        types=[GraphActionType.Stop, GraphActionType.AddNode, GraphActionType.AddEdge],
        keys=[None, "x", "y"],
    )
    return cat


def test_batch():
    cat = make_test_cat()
    assert (cat.batch[0] == torch.tensor([0, 1, 2])).all()
    assert (cat.batch[1] == torch.tensor([0, 0, 1, 1, 2, 2])).all()
    assert (cat.batch[2] == torch.tensor([0, 1])).all()


def test_slice():
    cat = make_test_cat()
    assert (cat.slice[0] == torch.tensor([0, 1, 2, 3])).all()
    assert (cat.slice[1] == torch.tensor([0, 2, 4, 6])).all()
    assert (cat.slice[2] == torch.tensor([0, 1, 2, 2])).all()


def test_logsoftmax():
    cat = make_test_cat()
    ls = cat.logsoftmax()
    # There are 3 graphs in the batch, so the total probability should be 3
    assert torch.isclose(sum([i.exp().sum() for i in ls]), torch.tensor(3.0))


def test_logsoftmax_grad():
    # Purposefully large values to test extremal behaviors
    logits = torch.tensor([[100, 101, -102, 95, 10, 20, 72]]).float()
    logits.requires_grad_(True)
    batch = Batch.from_data_list([Data(x=torch.ones((1, 10)), y=torch.ones((2, 6)))], follow_batch=["y"])
    cat = GraphActionCategorical(batch, [logits[:, :3], logits[:, 3:].reshape(2, 2)], [None, "y"], [None, None])
    cat._epsilon = 0
    gac_softmax = cat.logsoftmax()
    torch_softmax = torch.nn.functional.log_softmax(logits, dim=1)
    (grad_gac,) = torch.autograd.grad(gac_softmax[0].sum() + gac_softmax[1].sum(), logits, retain_graph=True)
    (grad_torch,) = torch.autograd.grad(torch_softmax.sum(), logits)
    assert torch.isclose(grad_gac, grad_torch).all()


def test_logsumexp():
    cat = make_test_cat()
    totals = torch.tensor(
        [
            # Plug in the arange values for each graph
            torch.logsumexp(torch.tensor([0.0, 0, 1, 2, 3, 4, 5, 6, 7, 0, 1, 2]), 0),
            torch.logsumexp(torch.tensor([1.0, 8, 9, 10, 11, 12, 13, 14, 15, 3, 4, 5]), 0),
            torch.logsumexp(torch.tensor([2.0, 16, 17, 18, 19, 20, 21, 22, 23]), 0),
        ]
    )
    assert torch.isclose(cat.logsumexp(), totals).all()


def test_logsumexp_grad():
    # Purposefully large values to test extremal behaviors
    logits = torch.tensor([[100, 101, -102, 95, 10, 20, 72]]).float()
    logits.requires_grad_(True)
    batch = Batch.from_data_list([Data(x=torch.ones((1, 10)), y=torch.ones((2, 6)))], follow_batch=["y"])
    cat = GraphActionCategorical(batch, [logits[:, :3], logits[:, 3:].reshape(2, 2)], [None, "y"], [None, None])
    cat._epsilon = 0
    (grad_gac,) = torch.autograd.grad(cat.logsumexp(), logits, retain_graph=True)
    (grad_torch,) = torch.autograd.grad(torch.logsumexp(logits, dim=1), logits)
    assert torch.isclose(grad_gac, grad_torch).all()


def test_sample():
    # Let's just make sure we can sample and compute logprobs without error
    cat = make_test_cat()
    actions = cat.sample()
    logprobs = cat.log_prob(actions)
    assert logprobs is not None


def test_argmax():
    cat = make_test_cat()
    # The AddNode logits has the most actions, and each graph has two rows each, so the argmax
    # should be 1,1,3 (1th action, AddNode, 1th row is larger due to arange, 3rd col is largest due
    # to arange)
    assert cat.argmax(cat.logits) == [(1, 1, 3), (1, 1, 3), (1, 1, 3)]


def test_log_prob():
    cat = make_test_cat()
    logprobs = cat.logsoftmax()
    actions = [[0, 0, 0], [2, 0, 2], [1, 1, 3]]
    correct_lp = torch.stack([logprobs[t][row + cat.slice[t][i], col] for i, (t, row, col) in enumerate(actions)])
    assert (cat.log_prob(actions) == correct_lp).all()

    actions = [[1, 1, 0], [1, 1, 1], [1, 1, 2], [1, 1, 3]]
    batch = torch.tensor([1, 1, 1, 1])
    correct_lp = torch.stack([logprobs[t][row + cat.slice[t][i], col] for i, (t, row, col) in zip(batch, actions)])
    assert (cat.log_prob(actions, batch=batch) == correct_lp).all()

    actions = [[0, 0, 0], [0, 0, 0], [0, 0, 0]]
    correct_lp = torch.arange(3)
    assert (cat.log_prob(actions, logprobs=cat.logits) == correct_lp).all()


def test_entropy():
    cat = make_test_cat()
    cat.entropy()
