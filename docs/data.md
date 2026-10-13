# The data

PEBR-Net is trained, validated and tested on a paired dataset of simulated borehole TEM profiles, available at
https://doi.org/10.57760/sciencedb.014t4. After downloading it, `python scripts/download_data.py --from <downloaded
archive or folder>` copies the six data files to `data/paired/` and checks their content against
`data/paired_content.json`.

## Files

```
data/paired/
  train_clean.csv   train_noisy.csv
  val_clean.csv     val_noisy.csv
  test_clean.csv    test_noisy.csv
```

Each profile has 61 stations from 0 to 300 m at 5 m spacing and 31 gates from 0 to 20 ms. Each row is one station
of one profile, and row *k* of a noisy file is the same station as row *k* of the matching clean file.

| column | content |
|---|---|
| `sample_id` | profile identifier |
| `depth_m` | station depth (m) |
| `t_<time>_s` | one column per gate; the header gives the gate time in seconds |

## Your own data

A paired dataset in the same layout can be used in its place: six files named `{train,val,test}_{clean,noisy}.csv`
with the columns above, and the same increasing gate times in all six files (`--paired_dir <folder>`).
