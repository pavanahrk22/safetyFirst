# Crime Data Analysis & Hotspot Prediction

A data analysis and visualization (DAV) project, Semester IV. It takes a crime incident CSV and produces:

- exploratory charts (crime types, monthly trends, hourly pattern, day × hour heatmap, arrest rates)
- a **hotspot ranking** of locations with Low / Medium / High risk levels
- a **Decision Tree vs Random Forest** model that predicts the *type* of a crime from when and where it happened
- an 18-month **baseline forecast** of monthly crime counts

## Dataset

This project does **not** ship with data and does **not** generate fake data. You must supply a real CSV.

Put it at `data/crime_data.csv` (or pass `--data path/to/file.csv`). It needs:

| Needed | Accepted column names |
|---|---|
| Date (ideally with time) | `Date` (or any column containing "date") |
| Crime type | `Crime_Type`, `Primary_Type`, `Type`, `Category` |
| Location / area | `Location_Description`, `Neighborhood`, `Community_Area_Name`, `Area`, `District`, `Beat`, `Ward`, `Location` |
| Arrest (optional) | `Arrest` (enables the arrest-rate chart) |

Column names are matched after trimming and replacing spaces with `_`, so `Primary Type` works. The column handling is built around the City of Chicago "Crimes" dataset layout.

> **Choosing the location column.** Auto-detection tries `Location_Description` first, which in the Chicago data describes the *kind* of place ("Street", "Residence"), not a geographic area. For real geographic hotspots use e.g. `--location-col District` or `--location-col Beat`.

## Setup

```bash
git clone https://github.com/pavanahrk22/safetyFirst.git
cd safetyFirst
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

## Usage

```bash
python -m crime_analysis --data data/crime_data.csv --location-col District
```

| Option | Default | Meaning |
|---|---|---|
| `--data` | `data/crime_data.csv` | input CSV |
| `--out` | `outputs` | folder for results |
| `--location-col` | auto-detect | column used as the hotspot area |
| `--max-model-rows` | `200000` | max rows sampled for model training (`0` = all rows) |
| `--seed` | `42` | random seed |

Results (all in `outputs/`): 10 PNG charts, `hotspots.csv`, `summary.json`.

## Project structure

```
crime_analysis/
  data.py       load + clean the CSV, detect columns
  features.py   time features; train-only location features
  eda.py        exploratory charts
  hotspots.py   hotspot ranking and risk levels
  model.py      Decision Tree vs Random Forest (+ majority-class baseline)
  forecast.py   trend + seasonality baseline forecast
  pipeline.py   command-line entry point
```

## Method notes

- **No leakage.** Location features (frequency, risk tier, id) are fitted on the training split only and then applied to the test split.
- **Baseline included.** Model accuracy is reported next to a majority-class baseline; crime types are imbalanced, so accuracy alone is misleading.
- **Forecast** is a least-squares quadratic trend plus yearly seasonality terms. It is a simple baseline. The shaded intervals come from in-sample residuals, so they are optimistic. A 6-month backtest MAPE is printed when there is enough history.

## Limitations

- The classifier only sees *time and place*, so it can't be expected to separate crime types well; compare it with the printed baseline before drawing conclusions.
- Hotspots are counts of recorded incidents, not rates: they do not account for population or reporting differences between areas.
- A partially covered first/last month is dropped from the forecast series automatically.

## Sample output

<!-- After running, copy a few charts from outputs/ into docs/images/ and embed them here, e.g.:
![Top hotspots](docs/images/chart_hotspots.png) -->

## License

MIT. See [LICENSE](LICENSE).

Author: Pavana H
