# Drought Severity Levels Using Time-Series Classification

This project aims to accurately predict drought severity by integrating temporal meteorological data with static soil and geospatial characteristics. By comparing traditional statistical benchmarks with modern deep learning architectures, we explore the most effective methods for mitigating the destructive impacts of drought through early prediction.

## Project Overview
Drought is a complex, multi-factor phenomenon. Unlike traditional methods that rely primarily on precipitation, our approach incorporates:
* **Meteorological Data:** 18 daily indicators (e.g., precipitation, temperature, humidity) from the NASA POWER Project.
* **Geospatial & Soil Data:** 32 static features (e.g., elevation, slope, land cover types) from the Harmonized World Soil Database (HWSD).
* **Classification:** Drought severity levels ranging from 0 (None) to 4 (D4), as defined by the U.S. Drought Monitor (USDM).

## Models & Architectures

### 1. ARIMAX (Baseline)
An **Autoregressive Integrated Moving Average with Exogenous variables** model used as a performance floor. It establishes how well linear feature importance can model drought using past values and exogenous meteorological regressors.

### 2. Long Short-Term Memory (LSTM)
A hybrid architecture designed to overcome the vanishing gradient problem in standard RNNs. 
* **Hybrid Processing:** Separates 180-day temporal sequences from static soil features.
* **Architecture:** 2-layer LSTM (128 hidden units) concatenated with static features before passing through a fully connected classification head.
* **Strength:** Best at balancing performance across imbalanced classes (highest Macro-F1).

### 3. 1D-Convolutional Neural Network (1D-CNN)
Utilizes a sliding filter to detect "signals" such as sudden temperature spikes or long-term precipitation decreases regardless of when they occur in the time series.
* **Architecture:** Convolutional layers with kernel size 2, MaxPooling, and global pooling to flatten outputs.
* **Strength:** Best at reducing the ordinal distance between predicted and actual labels (lowest MAE).

## Evaluation Metrics
* **Macro F1-Score:** Used to evaluate the model's ability to predict all drought categories, including rare high-severity events.
* **Mean Absolute Error (MAE):** Measures the average distance between predicted and actual labels, accounting for the ordinal nature of drought classes.

## Results
| Model | Macro-F1 | MAE |
| :--- | :---: | :---: |
| ARIMAX (Baseline) | 0.1884 | 0.8538 |
| **LSTM** | **0.3289** | 0.4144 |
| **1D-CNN** | 0.2967 | **0.3123** |

## Data Pipeline
* **Spatial Scope:** 3,109 US counties identified by FIPS codes.
* **Temporal Scope:** 180-day sequences.
* **Partitioning:**
  * **Training Set:** 2000–2009
  * **Validation Set:** 2010–2011 (Expanding window approach used)
  * **Testing Set:** 2012–2020

## Authors & Roles
* **Abrar:** Data preprocessing and ARIMAX baseline development.
* **Mousa Cheema:** LSTM model development and hybrid architecture design.
* **Khoi & James:** 1D-CNN model development and performance comparison.

## Repository Structure
The code for this project can be accessed at: [https://github.com/Abrar-135/drought-classification](https://github.com/Abrar-135/drought-classification)
