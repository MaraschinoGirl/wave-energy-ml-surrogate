

# README – WAVE_IRP_PROJECT

**Report Title:** Hybrid AI Pipeline for Surrogate Modelling in Wave Energy Farms

**Author:** Lesley-Ann Fenwick  

**University:** University of York, Department of Computer Science  

**Submission:** Independent Research Project (IRP), MSc Computer Science (AI)  

### OVERVIEW:

This project explores the development of a hybrid AI pipeline that integrates:

**LSTM (Long Short-Term Memory)** for inferring missing environmental variables, and **XGBoost** for high-performance power output prediction. Alongside **spatial layout modeling** of Wave Energy Converter (WEC) arrays.

The pipeline is benchmarked across geographically distinct sites (Perth and Sydney, Australia) and evaluated on predictive accuracy, generalization capacity, runtime efficiency, and layout sensitivity.



## Repository Structure
* WAVE_IRP_PROJECT/
* │
* ├── data/                     _# Original dataset files (CSV)_
* │   ├── WEC_Perth_49.csv
* │   ├── WEC_Perth_100.csv
* │   ├── WEC_Sydney_49.csv
* │   └── WEC_Sydney_100.csv
* │
* ├── figures/                  _# Auto-generated plots from the pipeline_
* │   ├── qw_inference_sydney.png
* │   ├── layout_hexbin_density.png
* │   └── ...
* │
* ├── src/                      _# Source code modules (modular architecture)_
* │   ├── config.py
* │   ├── data_loader.py
* │   ├── preprocessing.py
* │   ├── model_xgboost.py
* │   ├── model_lstm.py
* │   ├── features_layout.py
* │   ├── runtime_analysis.py
* │   ├── evaluate.py
* │   └── visuals.py
* │
* ├── main.py                   _# Master script to execute the full pipeline_
* ├── runlog.txt                _# Auto-logged run outputs_
* ├── README_WAVE_IRP_PROJECT.md
* ├── requirements.txt          _# Python library dependencies_
* └── exploratory_analysis.ipynb (optional Jupyter notebook for prototyping)


## Dataset Information

- **Source:** University of Adelaide / UCI ML Repository  
- **Name:** Large-scale Wave Energy Farm  
- **Instances:** 63,600  
- **Features:** 149 (including X, Y WEC coordinates, q-factor, total power output)  
- **License:** Creative Commons Attribution 4.0 (CC BY 4.0)  
- **Citation:**  
  Neshat, M., Alexander, B., Sergiienko, N., & Wagner, M. (2020). *Optimisation of large wave farms using a multi-strategy evolutionary framework.* GECCO.
- **Link:** https://archive.ics.uci.edu/dataset/882/large-scale+wave+energy+farm
---

## Getting Started

1. **Install dependencies:**

```bash
pip install -r requirements.txt
```

2. **Run the full pipeline:**

```bash
python main.py
```

This script handles:
- Data loading and preprocessing
- Feature engineering (spatial and temporal)
- Training LSTM for qW inference
- Training XGBoost for power prediction
- Evaluation and visual output

---

## Evaluation Metrics

- **MAE** (Mean Absolute Error)  
- **RMSE** (Root Mean Square Error)  
- **R²** (Coefficient of Determination)  
- **Runtime** (in seconds)  
- **Layout feature importance and clustering**  

All performance metrics are printed to console and visualized in the `figures/` directory.

---

## Artefact Notes

- **Preprocessed datasets were not saved as standalone CSVs**, but transformed in-memory using PyCharm scripts for each model stage (e.g., LSTM requires normalized qW with lag features).  
- The figures and performance logs reflect true model behavior under the designed pipeline conditions.
- Dataset integrity was preserved throughout by using unmodified CSVs from the source.

---

## Ethical Compliance

- Dataset contains no personal data or identifiers.  
- Project approved under the University of York Self-Assessment Ethics process.  
- All software components are open-source.  

---

## Acknowledgements

- Thanks to the University of Adelaide for dataset availability and to the UCI Machine Learning Repository.  
- This artefact is submitted in partial fulfillment of the IRP module at the University of York.

---

**Google Drive:** https://drive.google.com/drive/folders/1cKP2oHCBMfRek4rIFJLsE-zGziCSHHnf?usp=drive_link