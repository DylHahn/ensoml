# ENSO Forecasting from Deep Ocean Temperature Data

## Overview
This project forecasts El Niño 3.4 indices based on deep ocean temperature measurements in the Western Tropical Pacific. It leverages both traditional machine learning (linear regression) and deep learning models (LSTM, NHITS) to predict ENSO-related anomalies.

The project includes two main workflows:
- **Linear Models:** Simple regression models using GODAS reanalysis data.
- **Neural Forecast Models:** Sequence models trained on surface temperature anomalies.

## Project Structure
```
.
├── data/                # Preprocessed GODAS climatology and anomaly files
├── figures/             # Saved prediction plots
├── stats/               # Output statistics and model predictions
├── enso_models.py       # Main script with modeling classes
└── README.md            # Project documentation
```

## Requirements
- Python 3.8+
- Libraries:
  - `numpy`
  - `pandas`
  - `scipy`
  - `xarray`
  - `sklearn`
  - `matplotlib`
  - `plotly`
  - `neuralforecast`

Install dependencies with:
```bash
pip install -r requirements.txt
```

(You'll need to create `requirements.txt` or install manually.)

## Data
- GODAS climatology files: gridded ocean temperatures.
- Precomputed ENSO 3.4 surface anomalies.

Make sure `data/` contains:
- `godasClimatologyData_{depth}m.nc` (NetCDF files)
- `movingAverageAnomalies{depth}m.txt` (Text files)

## How It Works

### Linear Regression
- **Input:** Deep ocean temperatures at various depths.
- **Output:** Forecasted surface ENSO 3.4 anomalies.
- Trains using 1980-1995 data, validates on 1997-2006.
- Outputs:
  - Correlation and RMSE scores.
  - Time series plots of predictions vs ground truth.

Run:
```python
from enso_models import EnsoLinearModel
model = EnsoLinearModel()
model.run_all_scenarios()
```

### Neural Forecasting
- **Input:** Surface anomalies (5m depth).
- **Output:** Multi-month forecasts of ENSO 3.4 index.
- Uses `neuralforecast` models (e.g., LSTM, NHITS).

Run:
```python
from enso_models import EnsoNeuralModel
model = EnsoNeuralModel()
model.run_nf_scenario()
model.show_nf_plot()
```

## Notes
- Linear models require manually selected non-NaN grid points.
- Neural models currently only use surface anomalies; future updates could incorporate 3D fields.
- Hyperparameters (e.g., horizon length, hidden sizes) can be tuned.

## References
- [NeuralForecast GitHub](https://github.com/Nixtla/neuralforecast)
- [HuggingFace Time Series Transformers](https://huggingface.co/blog/time-series-transformers)
- [Autoformer Paper](https://huggingface.co/blog/autoformer)
