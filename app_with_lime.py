from flask import Flask, request, jsonify
import joblib
import os
import pandas as pd
from scipy.sparse import hstack
import re
import html
import threading
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import lime
import lime.lime_tabular
import numpy as np

app = Flask(__name__)

# Loading the Model and tools
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

try:
    vectorizer = joblib.load(os.path.join(BASE_DIR, 'TF-IDF_vectorizer.pkl'))
    encoder = joblib.load(os.path.join(BASE_DIR, 'onehot_encoder.pkl'))
    model = joblib.load(os.path.join(BASE_DIR, 'best_svm_tuned_model.pkl'))
    print('✅ All models and tools loaded successfully!')
except Exception as e:
    print(f'❌ Error loading files: {e}')
    raise

# Text Cleaning Function
stop_words = set(stopwords.words('english'))
lem = WordNetLemmatizer()

def text_clean(text):
    if not isinstance(text, str):
        text = str(text)
    text = html.unescape(text)
    text = text.lower()
    text = re.sub(r'http\S+', '', text)
    text = re.sub(r'<.*?>', '', text)
    text = re.sub(r'[^\w\s$]', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    words = text.split()
    words = [word for word in words if word not in stop_words]
    words = [lem.lemmatize(word, pos='v') for word in words]
    return ' '.join(words)

# Column Definitions
text_columns = ['title', 'company_profile', 'description', 'requirements', 'benefits']
categorical_columns = ['department', 'employment_type', 'required_experience',
                       'required_education', 'industry', 'function', 'country']

# Loading Background Data for LIME 
# Training data for LIME to work
X_train = joblib.load(os.path.join(BASE_DIR, 'X_train.pkl'))   # We will keep this file locally

feature_names = (vectorizer.get_feature_names_out().tolist() + 
                 encoder.get_feature_names_out(categorical_columns).tolist())

explainer = lime.lime_tabular.LimeTabularExplainer(
    X_train[:1000].toarray(),
    feature_names=feature_names,
    class_names=['Real', 'Fake'],
    mode='classification'
)
print('✅ LIME Explainer is ready!')

# Prediction + Explanation Function
def predict_with_explanation(new_job_dict, threshold=0.5):
    # Clean and transform input
    for col in text_columns:
        new_job_dict[col] = new_job_dict.get(col, 'unspecified')
        new_job_dict[col] = text_clean(new_job_dict[col])
    
    combined_text = ' '.join(new_job_dict[col] for col in text_columns)
    
    X_text_new = vectorizer.transform([combined_text])
    cat_df = pd.DataFrame({col: [new_job_dict.get(col, 'unspecified')] for col in categorical_columns})
    X_cat_new = encoder.transform(cat_df)
    X_new = hstack([X_text_new, X_cat_new]).toarray()[0]
    
    # Prediction
    prob = model.predict_proba(X_new.reshape(1, -1))[0][1]
    result = 'Fake' if prob >= threshold else 'Real'
    
    # LIME Explanation
    exp = explainer.explain_instance(
        X_new, 
        model.predict_proba, 
        num_features=10
    )
    
    # Get top reasons
    reasons = []
    for feature, weight in exp.as_list():
        if weight > 0:
            reasons.append(f"{feature} (supports Fake)")
        else:
            reasons.append(f"{feature} (supports Real)")
    
    return {
        "prediction": result,
        "fake_probability_percent": round(prob * 100, 2),
        "explanation": reasons[:8]   # Top 8 reasons
    }

# Flask Route
@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()
        if not data:
            return jsonify({'error': 'No JSON data provided'}), 400
        
        result = predict_with_explanation(data)
        return jsonify(result)
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# Run
if __name__ == '__main__':
    print('🚀 Flask API with LIME is starting...')
    print('Test URL: http://127.0.0.1:5001/predict')   # Using port 5001 to avoid conflict
    app.run(debug=False, host='0.0.0.0', port=5001)