# Supervised-Learning-Hit-Or-Not
Interactive link to ReviewNB https://app.reviewnb.com/susmitaSanyal/Supervised-Learning-Hit-Or-Not/blob/main/HitSongPrediction%20(1).ipynb/file
# Predicting Song Popularity on Spotify Using Supervised Learning

> A machine learning project to classify whether a track will be a **hit** or **non‑hit** from Spotify audio features.

Author: Susmita Sanyal
Original notebook: [https://github.com/susmitaSanyal/Supervised-Learning-Hit-Or-Not](https://github.com/susmitaSanyal/Supervised-Learning-Hit-Or-Not)

## Table of Contents

* [Overview](#overview)
* [Dataset](#dataset)
* [Feature Dictionary](#feature-dictionary)
* [Methodology](#methodology)
* [Results](#results)
* [Project Structure](#project-structure)
* [Setup](#setup)
* [Reproduce the Notebook](#reproduce-the-notebook)
* [Notes & Limitations](#notes--limitations)
* [References](#references)
* [License](#license)

## Overview

Objective:
The goal of this project is to develop a classification model that predicts whether a given song is likely to be a “hit” (popular) or “non-hit” based on its musical and acoustic characteristics available on Spotify.

Formally, given a set of audio features (e.g., danceability, energy, valence, loudness, tempo, key, mode, etc.) extracted from track metadata via the Spotify Web API, build a supervised learning model that maps these features to a binary target variable indicating whether a track meets the criteria of a “hit.”

The label can be defined based on a threshold of the popularity metric provided by Spotify, which is a proprietary score ranging from 0 to 100 that reflects a track’s overall popularity on the platform based on play counts and recency.

The final model should learn to classify songs into popularity categories such as:

1 (Popular / Hit)

0 (Unpopular / Non-hit)

## Dataset

Dataset: Ultimate Spotify Tracks Dataset

Repository: [Kaggle Dataset – Ultimate Spotify Tracks DB](https://www.kaggle.com/datasets/zaheenhamidani/ultimate-spotify-tracks-db)

Author: Zaheen Hamidani (Kaggle Contributor)

Data Format: `CSV`

**Collection Method:**

The dataset aggregates data using the Spotify Web API, which provides audio features for over 232,000 tracks. The data includes various song attributes and popularity scores ingested over time periods, collected by querying the Spotify API via the spotifyr or spotipy libraries.

**Use Cases:**

* Academic and educational purposes to teach classification techniques in machine learning.
* Exploration of audio features and their relationship to track popularity.
* Real-world applications in music recommendation, playlist curation, marketing optimization, and trend forecasting.

## Feature Dictionary

Below is a concise dictionary of the most salient Spotify audio features used in the models (abbreviated).

Key audio features provided by Spotify’s Web API (also present in the dataset) include:

| Feature            | Description                                  | Type          |
| ------------------ | -------------------------------------------- | ------------- |
| `danceability`     | How suitable a track is for dancing          | Numeric (0–1) |
| `energy`           | Perceived intensity and activity measure     | Numeric (0–1) |
| `loudness`         | Overall loudness in decibels (dB)            | Numeric       |
| `speechiness`      | Presence of spoken words                     | Numeric (0–1) |
| `acousticness`     | Confidence whether a track is acoustic       | Numeric (0–1) |
| `instrumentalness` | Extent to which the track lacks vocals       | Numeric (0–1) |
| `liveness`         | Presence of an audience in the recording     | Numeric (0–1) |
| `valence`          | Musical positiveness conveyed by a track     | Numeric (0–1) |
| `tempo`            | Estimated beats per minute (BPM)             | Numeric       |
| `key`              | Estimated overall key (Pitch Class notation) | Categorical   |
| `mode`             | Major (1) or minor (0)                       | Binary        |
| `time_signature`   | Estimated number of beats per bar            | Categorical   |
| `duration_ms`      | Track duration                               | Numeric       |
| `popularity`       | Popularity score (0–100)                     | Numeric       |

> See the notebook for additional identifiers (e.g., `track_id`, `artist`, `genre`) and engineered features.

## Methodology

* **Task**: Binary classification (Hit = 1, Non‑hit = 0).
* **Model families explored**: Logistic Regression, k‑Nearest Neighbors, Random Forest, Gradient Boosting, XGBoost, Naïve Bayes.
* **Evaluation**: Stratified train/validation split with Accuracy / Precision / Recall; additional qualitative checks via threshold analysis and feature importance where applicable.
* **Preprocessing**: Numeric scaling, handling of class imbalance (if present), removal of identifier columns, and basic data hygiene (deduplication, null handling).

> See the accompanying notebook for EDA (distribution plots, correlation heatmaps), modeling code, and ablation notes.

## Results

A brief summary of findings from the notebook:

* Tree‑based ensembles (e.g., **Random Forest**, **Gradient Boosting**, **XGBoost**) generally produced the strongest performance among tested baselines.
* Linear baselines (e.g., **Logistic Regression**) offered competitive accuracy with high interpretability and lower computational cost.
* Threshold tuning matters: overly aggressive cutoffs can degrade precision/recall trade‑offs.

This analysis demonstrated that effective predictive modeling requires balancing interpretability and performance. Tree‑based ensembles showed promise in capturing non‑linear interactions among audio features, often outperforming simpler linear models. However, logistic regression served as a transparent and robust baseline, providing a straightforward interpretation of feature effects and achieving relatively strong results with minimal tuning. The exploration also highlighted the importance of appropriate thresholding and careful feature selection; overly strict cutoffs can introduce noise rather than precision. Moreover, computational efficiency and resource constraints influenced model selection, suggesting that for production contexts, a well-regularized linear model or a moderately tuned ensemble might achieve comparable performance with lower computational costs.

## Project Structure

```
.
├── notebooks/
│   └── HitSongPrediction.ipynb        # Source notebook for this README
├── data/
│   ├── raw/                           # Raw CSV from Kaggle (not committed)
│   └── processed/                     # Cleaned/engineered datasets
├── src/
│   ├── features/                      # Feature engineering scripts
│   ├── models/                        # Training/inference utilities
│   └── visualization/                 # Plot helpers
├── reports/
│   └── figures/                       # Exported charts
├── LICENSE
└── README.md
```

## Setup

1. Create and activate a Python environment (3.9+ recommended).
2. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```

   Suggested packages (if you’re reconstructing `requirements.txt`):

   ```
   numpy
   pandas
   scikit-learn
   xgboost
   matplotlib
   seaborn
   jupyter
   ```
3. (Optional) Set environment variables for data paths:

   ```bash
   export DATA_DIR=./data
   ```

## Reproduce the Notebook

1. Download the dataset from Kaggle (see link below) and place the CSVs in `data/raw/`.
2. Launch Jupyter:

   ```bash
   jupyter lab
   ```
3. Open `notebooks/HitSongPrediction.ipynb` and run all cells.

## Notes & Limitations

* This is a **classification framing** of popularity; real‑world “hit” prediction is influenced by promotion, playlisting, and socio‑cultural factors beyond audio features alone.
* Class balance and label definition (what counts as a “hit”) materially impact reported metrics.
* Model performance can vary by time period and market; consider time‑based validation for production use.

## References

* Kaggle: *Ultimate Spotify Tracks DB* by Zaheen Hamidani.
* Spotify Web API Documentation.
* Original notebook by Susmita Sanyal (credited above).

## License

If you publish this repository publicly, pick a license appropriate to your needs (e.g., MIT, Apache‑2.0). If this project derives from third‑party data/code, **respect their licenses and terms of use**.
