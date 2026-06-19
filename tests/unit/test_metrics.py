from ncf_recommender.evaluation.metrics import hit_rate_at_k, mrr_at_k, ndcg_at_k, recall_at_k


def test_hit_rate_at_k() -> None:
    assert hit_rate_at_k([2, 5, 1], gt_item=5) == 1.0
    assert hit_rate_at_k([2, 3, 1], gt_item=5) == 0.0


def test_ndcg_at_k() -> None:
    assert ndcg_at_k([5, 2, 1], gt_item=5) == 1.0
    assert ndcg_at_k([2, 5, 1], gt_item=5) > 0.0
    assert ndcg_at_k([2, 3, 1], gt_item=5) == 0.0


def test_mrr_at_k() -> None:
    assert mrr_at_k([5, 2, 1], gt_item=5) == 1.0
    assert mrr_at_k([2, 5, 1], gt_item=5) == 0.5
    assert mrr_at_k([2, 3, 1], gt_item=5) == 0.0


def test_recall_at_k_single_positive() -> None:
    assert recall_at_k([2, 5, 1], ground_truth_items={5}) == 1.0
    assert recall_at_k([2, 3, 1], ground_truth_items={5}) == 0.0
