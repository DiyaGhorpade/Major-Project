# Plant stress classifier

Train and evaluate three classifiers: Logistic Regression, XGBoost and a PyTorch
MLP. They predict Healthy, Moderate Stress or High Stress from the bundled
`data/plant_health_data.csv` dataset.

This is an offline training workflow. It requires features beyond those collected
by the ESP32 sketches, so the receiver's CSV cannot be passed directly into it.
Forecasting, plant recommendations and live inference are not provided by these scripts.

## 1. Create the training environment

Use the separate Conda environment defined here. From the repository root:

```sh
cd plant_stress
conda env create -f environment.yml
conda activate plant-stress-step1
python verify_env.py
```

Keep this terminal in **`plant_stress/`** for all commands below. The environment
uses Python 3.10 and its own package versions; do not install it into the root
Python 3.12 environment.

`verify_env.py` prints package versions, CUDA availability and detected GPU.
The checked-in configuration requests CUDA for XGBoost. Before running on a
CPU-only machine, set `XGB_PARAMS["device"]` to `"cpu"` in `src/config.py`;
PyTorch selects its device automatically through `DEVICE`.

The previous GPU setup used PyTorch 2.7.0 with CUDA 12.8. That optional wheel
installation command is:

```sh
python -m pip install torch==2.7.0 torchvision==0.22.0 torchaudio==2.7.0 --index-url https://download.pytorch.org/whl/cu128
```

Use it only for a compatible GPU/driver environment. Jetson-specific packages
and deployment have not been verified by this setup.

## 2. Check the dataset and configuration

The input file is `data/plant_health_data.csv`, included in this repository.
`src/config.py` defines the input path, feature columns, label mapping, train/test
split, model settings and output directories. Keep the dataset and configured
feature names aligned.

The dataset is synthetic. Evaluation on it measures performance on that dataset;
it does not establish accuracy on the commissioned plants or sensors.

## 3. Run the scripts in order

```sh
python src/preprocess.py
python src/train.py
python src/evaluate.py
```

| Script | Result |
| --- | --- |
| `verify_env.py` | Prints installed versions and device availability. |
| `src/preprocess.py` | Loads data, engineers features, splits/scales readings and saves `outputs/models/scaler.joblib`. |
| `src/train.py` | Repeats preprocessing, trains the three models and saves model artifacts in `outputs/models/`. |
| `src/evaluate.py` | Repeats preprocessing, loads trained models and writes classification reports and evaluation/SHAP plots. Run training first. |

Reports go to `outputs/reports/` and plots to `outputs/plots/`. Rerunning the
scripts writes to the same artifact paths; preserve any results you need first.
There is no implemented export script or deployment command in this folder.
