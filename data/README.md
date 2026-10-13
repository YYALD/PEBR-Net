# data/

The paired dataset is distributed separately: https://doi.org/10.57760/sciencedb.014t4. This folder holds what is
needed to check it:

- `sources.json`: where the dataset and the trained checkpoint are published;
- `paired_content.json`: the content of the six data files (rows, profiles, gate times, depths, value sums);
- `paired.sha256`: the SHA-256 of one copy of them.

```bash
python scripts/download_data.py --from PATH  # the downloaded archive or folder: unpack and check to data/paired/
python scripts/download_data.py --verify     # check data/paired/ only
```

[../docs/data.md](../docs/data.md) describes the dataset.
