# Migration Notes

## Script mapping
- `GMF.py` -> `ncf_recommender.models.gmf` + `ncf_recommender.cli.train`
- `MLP.py` -> `ncf_recommender.models.mlp` + `ncf_recommender.cli.train`
- `NeuMF.py` -> `ncf_recommender.models.neumf` + `ncf_recommender.cli.train`
- `Dataset.py` -> `ncf_recommender.data.datasets`
- `evaluate.py` -> `ncf_recommender.evaluation.metrics` + `ncf_recommender.evaluation.protocols`

## Behavioral parity targets
- Leave-one-out evaluation
- Top-K ranking metrics (HR/NDCG) compatible with legacy protocol
- Optional pretraining GMF + MLP then warm-start NeuMF

## Config mapping
- Legacy `python GMF.py ...` -> `ncf-train --config-name train_gmf`
- Legacy `python MLP.py ...` -> `ncf-train --config-name train_mlp`
- Legacy `python NeuMF.py ...` -> `ncf-train --config-name train_neumf`
- Legacy NeuMF with `--mf_pretrain ... --mlp_pretrain ...` -> `ncf-train --config-name train_neumf_pretrained`

## Argument mapping examples
- Legacy `--num_factors` -> `model.embedding_dim` (GMF) or `model.mf_dim` (NeuMF)
- Legacy `--layers` -> `model.layers` (MLP) or `model.mlp_layers` (NeuMF)
- Legacy `--num_neg` -> `train.num_negatives`
- Legacy `--lr` -> `train.learning_rate`
- Legacy `--epochs` -> `train.epochs`
- Legacy `--batch_size` -> `train.batch_size`

## Command conversion
Legacy GMF:
```bash
python GMF.py --dataset ml-1m --epochs 20 --batch_size 256 --num_factors 8 --num_neg 4 --lr 0.001
```
Modern:
```bash
ncf-train --config-name train_gmf --set data.name=ml-1m --set train.epochs=20 --set train.batch_size=256 --set model.embedding_dim=8 --set train.num_negatives=4 --set train.learning_rate=0.001
```

Legacy NeuMF pretrained:
```bash
python NeuMF.py ... --mf_pretrain path_gmf.h5 --mlp_pretrain path_mlp.h5
```
Modern:
```bash
ncf-train --config-name train_neumf_pretrained --set model.pretrain.gmf_checkpoint=path_gmf.pt --set model.pretrain.mlp_checkpoint=path_mlp.pt
```

## New production commands
- Benchmark: `ncf-benchmark --config-name eval_gmf`
- Export: `ncf-export --config-name eval_neumf --export-format torchscript --output-path artifacts/exports/neumf.ts`
- Serve API: `ncf-serve --host 0.0.0.0 --port 8000`
