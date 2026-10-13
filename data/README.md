# Data setup

The dataset is intentionally excluded from this repository.

1. Obtain MicroLens-1M MMCTR from the [official dataset page](https://recsys.westlake.edu.cn/MicroLens_1M_MMCTR) or the links provided by the [challenge organizers](https://erel-mir.github.io/challenge/mmctr-track2/).
2. Review and follow the dataset's original access and license terms.
3. Place the files in the layout below.

```text
data/
├── MicroLens_1M_x1/
│   ├── train.parquet
│   ├── valid.parquet
│   ├── test.parquet
│   └── item_info.parquet
├── item_emb.parquet
├── item_feature.parquet
└── item_seq.parquet
```

Run `python scripts/check_data.py` from the repository root before training. Generated ablation embeddings are written under `data/MicroLens_1M_x1/ablation/` and remain ignored by Git.
