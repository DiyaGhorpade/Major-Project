# Exploratory data analysis

All analysis notebooks, datasets, generated results, and supporting documents
live here, separate from `../hardware-code/` and `../sensor_data_reciever/`.

| Folder | Contents |
| --- | --- |
| `notebooks/` | Agricultural, indoor-plant, and greenhouse analysis notebooks |
| `data/` | Greenhouse growth metrics and plant health monitoring datasets |
| `scripts/` | Standalone greenhouse clustering diagnostics |
| `outputs/` | Generated clustering CSV metrics, JSON diagnostics, and PNG plot |
| `reports/` | Notebook flow analysis and greenhouse EDA audit |
| `docs/` | Supporting Word documents |

## Notebooks

Open notebooks from `notebooks/`. The agricultural and greenhouse notebooks
locate `eda/data/` from the repository root or any directory inside the repository.
Their existing saved outputs are retained; moving files does not rerun analyses.

`mp_1_indoor_plants (1).ipynb` still uses Google Colab and loads
`Indoor_Plant_Health_and_Growth_Factors.csv` from Google Drive. That dataset is
not included in this repository. Use its existing Colab workflow or supply the
dataset and update the loading cells before running it locally.

The notebooks use packages including NumPy, pandas, Matplotlib, seaborn, and
scikit-learn; the indoor-plants notebook also uses the Colab environment.

## Standalone diagnostics

From the repository root:

```powershell
python eda/scripts/clustering_root_cause_diagnostics.py
```

The script reads `data/Greenhouse Plant Growth Metrics.csv` and writes its
artifacts into `outputs/`, independently of the current working directory.
It evaluates many K-Means fits and full-dataset silhouette scores, so a run can
take substantial time.

The reports are historical reviews and may describe earlier notebook versions.
