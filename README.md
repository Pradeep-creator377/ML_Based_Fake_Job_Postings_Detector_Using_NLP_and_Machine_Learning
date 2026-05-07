# ML-Based Fake Job Postings Detector
A machine learning project that detects fake job postings using NLP techniques and an SVM classifier.

## Project Overview
- This system reads a job posting and predicts whether it is Real or Fake. 
- I used text cleaning, TF-IDF features, trained multiple models, and deployed it using Flask API. 
- I also added LIME to explain why the model made a prediction.

## Features
- Text Preprocessing with TF-IDF Vectorization
- Models trained: Logistic Regression, SVM, XGBoost
- Best Model: SVM (good precision)
- Flask REST API for real-time prediction
- LIME Explainability (shows reasons behind prediction)

## Files in this Repository
- ML_Based_Fake_Job_Postings_Detector_Using_NLP_and_Machine_Learning.ipynb → Main Jupyter Notebook
- app.py → Fast Flask API (Prediction only)
- app_with_lime.py → Flask API with LIME Explanation
- TF-IDF_vectorizer.pkl
- onehot_encoder.pkl
- best_svm_tuned_model.pkl
- requirements.txt → List of all dependencies to install

## Dataset
- The dataset used in this project is publicly available.
- Download link: [Fake Job Postings Dataset](https://www.kaggle.com/datasets/shivamb/real-or-fake-fake-jobposting-prediction)
- File name: `fake_job_postings.csv`

## How to Run the Project

### 1. Install Requirements

```bash
pip install -r requirements.txt
```

### 2. Using Jupyter Notebook 

- Open the notebook ML_Based_Fake_Job_Postings_Detector_Using_NLP_and_Machine_Learning.ipynb
- Run all cells from top to bottom
- You can see EDA, training, evaluation, and LIME explanations

### 3. Running the Flask API

**Option A: Fast Prediction (Recommended)**
- Run this file:
  ```bash
  python app.py
  ```
  Test URL: http://127.0.0.1:5000/predict

**Option B: Prediction + Explanation**
- Run this file:
  ```bash
  python app_with_lime.py
  ```
  Test URL: http://127.0.0.1:5001/predict

Note: This may take some time (10-30 seconds) because it also generates LIME explanation.

## Technologies Used
- Python, Pandas, Scikit-learn, XGBoost
- NLTK (text cleaning)
- TF-IDF Vectorizer
- Flask (API)
- LIME (Explainability)

## Note
- app.py is fast but gives only prediction.
- app_with_lime.py gives both prediction and reasons (why the model decided Real or Fake).

## Final Model
- Among the models trained (Logistic Regression, SVM, XGBoost), **SVM** achieved the best F1 score and is used as the default final model in this project.
- However, for flexibility, the tuned **Logistic Regression** and **XGBoost** models are also included (`best_lr_tuned_model.pkl`, `best_xgb_tuned_model.pkl`).  
- Users can load and test these models if they want to compare performance or explore alternative approaches.

**Thank You**

