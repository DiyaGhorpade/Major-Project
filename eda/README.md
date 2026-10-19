# Dataset analysis

Use the notebooks to explore the supplied datasets, or run the standalone
script to reproduce greenhouse clustering diagnostics.

## Setup

From the repository root, create/activate the Python 3.12 environment using the
[root setup instructions](../README.md#python-312-setup), then install:

```sh
python -m pip install -r requirements.txt
python -m ipykernel install --user --name major-project-py312 --display-name "Python 3.12 (Major-Project)"
```

## Run a notebook

Open a notebook in VS Code or another notebook editor and select the
`Python 3.12 (Major-Project)` kernel.

| Notebook in `notebooks/` | Input |
| --- | --- |
| `agricultural-dataset.ipynb` | Supplied data under `eda/data/`. |
| `greenhouse-plant-growth-metrics-eda.ipynb` | Supplied greenhouse data under `eda/data/`. |
| `mp_1_indoor_plants (1).ipynb` | Google Colab/Drive workflow; its indoor-plant dataset is not included. Supply the dataset and adjust loading cells to run locally. |

The agricultural and greenhouse notebooks locate the data from the repository
root or a directory inside the repository. Run cells in order. Existing saved
outputs are previous results; rerunning cells regenerates them.

## Run clustering diagnostics

With the root environment activated, run from the repository root:

```sh
python eda/scripts/clustering_root_cause_diagnostics.py
```

Input: `eda/data/Greenhouse Plant Growth Metrics.csv`.
Outputs: metrics CSVs, diagnostic JSON and a PNG plot in `eda/outputs/`.
The script evaluates many K-Means fits and silhouette scores, so it can take
substantial time. It writes to the same output locations on subsequent runs.

Supporting analysis reviews are in `reports/`, and documents are in `docs/`.
This analysis does not start data collection or read the Jetson receiver CSV automatically.
