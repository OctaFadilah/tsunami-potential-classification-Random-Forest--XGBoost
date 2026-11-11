# Developed a predictive machine learning model to classify tsunami-triggering earthquakes in Japan (2000-2025) to support disaster mitigation systems.
![Python](https://img.shields.io/badge/Python-3.x-blue.svg)
![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-Latest-orange.svg)
![XGBoost](https://img.shields.io/badge/XGBoost-Latest-green.svg)
![Machine Learning](https://img.shields.io/badge/Machine%20Learning-Classification-red.svg)

---

## Overview
This project is a geophysical computational study comparing the performance of machine learning algorithms (Random Forest and XGBoost) in classifying tsunami potential based on historical earthquake data in Japan (2000-2025). The novelty of this project lies in the integration of fault mechanism parameters (fault type) as an additional feature to improve the precision of tsunami early warning predictions and reduce the number of false alarms.

---

## 📸 Outputs Project Demonstration

<p align="center">
  <img src="outputs/GIF outputs.gif" width="650">
</p>
<p align="center">
  <em>Figure 1: Model Performance Demo.</em>
</p>

---

## Research Objectives
* To analyze and compare the reliability of Random Forest and XGBoost in detecting tsunami-triggering earthquakes.
* To evaluate the impact of adding fault type parameters on classification performance compared to using only basic seismic parameters (magnitude and spatial data).
* To minimize false negatives (missed tsunamis) and false positives (false alarms) through Confusion Matrix analysis and threshold optimization on the Precision-Recall curve.

---

## Dataset & Features
* **Data Sources:** Earthquake data was acquired from the *National Research Institute for Earth Science and Disaster Resilience* (NIED) Japan, while the tsunami catalog was integrated from the *National Oceanic and Atmospheric Administration* (NOAA).
* **Primary Features:** Magnitude (Mw), Depth (km), Longitude, Latitude.
* **Additional Feature (Main Focus):** Fault Type (Normal, Reverse, Strike-Slip), derived and converted from the rake angle value in the focal mechanism data.
* **Data Challenge:** The dataset has an extreme class imbalance of 1:22 between tsunami and non-tsunami events. This was handled at the algorithmic level using the `class_weight='balanced'` parameter for Random Forest and `scale_pos_weight` for XGBoost.

---

## Technologies & Tools
* **Programming Language:** Python
* **Machine Learning & Evaluation:** Scikit-Learn (Random Forest, GridSearchCV, evaluation metrics), XGBoost
* **Data Processing:** Pandas, NumPy
* **Visualization & Geospatial Mapping:** Matplotlib, Seaborn, Basemap, Global Land Mask

---

## Methodology
1. **Pre-processing & Integration:** Spatial coordinates were filtered specifically within Japan's borders, applying empirical thresholds of magnitude $\ge 5.3$ Mw and depth $\le 68$ km. Earthquake and tsunami data were then merged to generate classification labels.
2. **Experimental Design:** The data was partitioned into two testing scenarios: Dataset 1 (basic seismic parameters only) and Dataset 2 (integrated with fault type features).
3. **Training & Tuning:** The dataset was split into 80% training data and 20% testing data. Optimal hyperparameters were searched automatically using GridSearchCV to prevent model overfitting.
4. **Threshold Optimization:** The probability threshold was manually calibrated using the Precision-Recall curve to ensure the sensitivity rate (Recall) reached 100% without compromise.

---

## Key Results & Conclusion
* **Perfect Tsunami Detection:** After threshold optimization, both Random Forest and XGBoost successfully detected all actual tsunami events (Recall 1.00 / 0 False Negatives).
* **XGBoost Superiority:** Overall, the XGBoost model slightly outperformed Random Forest with an AUC value of 0.986 compared to Random Forest's 0.985.
* **Impact of Fault Type (Feature Importance):**
  * The XGBoost model successfully utilized the fault type parameter (specifically the reverse fault) as a crucial indicator in predicting seafloor deformation.
  * Integrating this feature improved XGBoost's Precision from 0.6000 to 0.6429, and its F1-score from 0.7500 to 0.7826.
  * As a result, the number of false alarms (False Positives) on XGBoost was reduced from 6 to 5 events.
  * Conversely, Random Forest's performance remained stagnant because its bagging and majority voting methods were heavily dominated by the Magnitude feature, causing the fault type information to be ignored.
* **Final Conclusion:** The sequential gradient boosting mechanism in the XGBoost architecture proved to be more precise, analytical, and adaptive as a computational model for tsunami early warning and disaster mitigation systems.

---

## Getting Started / Installation

1. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```
2. **Run the notebooks:** Open the `Main.py` files using Jupyter Notebook, JupyterLab, or upload them directly to Google Colab.

---

## Reference
Fadilah, O. (2026). *Analisis Perbandingan Random Forest dan XGBoost dalam Klasifikasi Potensi Tsunami Berdasarkan Data Gempa Bumi Tahun 2000-2025 di Jepang*. Universitas Pembangunan Nasional "Veteran" Yogyakarta.
