import torch

from ncf_recommender.models.gmf import GMF
from ncf_recommender.models.mlp import MLP
from ncf_recommender.models.neumf import NeuMF
from ncf_recommender.training.trainer import _load_pretrained_neumf_weights


def test_neumf_pretrain_weight_loading() -> None:
    gmf = GMF(num_users=10, num_items=20, embedding_dim=8)
    mlp = MLP(num_users=10, num_items=20, layer_sizes=[64, 32, 16, 8])
    neumf = NeuMF(num_users=10, num_items=20, mf_dim=8, mlp_layers=[64, 32, 16, 8])

    with torch.no_grad():
        gmf.user_embedding.weight.fill_(0.1)
        gmf.item_embedding.weight.fill_(0.2)
        mlp.user_embedding.weight.fill_(0.3)
        mlp.item_embedding.weight.fill_(0.4)

    _load_pretrained_neumf_weights(neumf, gmf, mlp)

    assert torch.allclose(neumf.mf_user_embedding.weight, gmf.user_embedding.weight)
    assert torch.allclose(neumf.mf_item_embedding.weight, gmf.item_embedding.weight)
    assert torch.allclose(neumf.mlp_user_embedding.weight, mlp.user_embedding.weight)
    assert torch.allclose(neumf.mlp_item_embedding.weight, mlp.item_embedding.weight)
