import torch

from ncf_recommender.models.gmf import GMF
from ncf_recommender.models.mlp import MLP
from ncf_recommender.models.neumf import NeuMF


def test_gmf_forward_shape() -> None:
    model = GMF(num_users=100, num_items=200, embedding_dim=16)
    scores = model(torch.tensor([1, 2]), torch.tensor([3, 4]))
    assert scores.shape == (2,)


def test_mlp_forward_shape() -> None:
    model = MLP(num_users=100, num_items=200, layer_sizes=[64, 32, 16, 8])
    scores = model(torch.tensor([1, 2]), torch.tensor([3, 4]))
    assert scores.shape == (2,)


def test_neumf_forward_shape() -> None:
    model = NeuMF(num_users=100, num_items=200, mf_dim=8, mlp_layers=[64, 32, 16, 8])
    scores = model(torch.tensor([1, 2]), torch.tensor([3, 4]))
    assert scores.shape == (2,)
