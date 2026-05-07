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

    # Initializing Flask App 
    app = Flask(__name__)

    # Loading Saved Models & Tools 
    project_fold = os.path.dirname(os.path.abspath(__file__))

    try:
        vectorizer = joblib.load(os.path.join(project_fold, 'TF-IDF_vectorizer.pkl'))
        encoder = joblib.load(os.path.join(project_fold, 'onehot_encoder.pkl'))
        model = joblib.load(os.path.join(project_fold, 'best_svm_tuned_model.pkl'))
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

    # Prediction Function 
    def predict_job(new_job_dict, threshold=0.5):
        # Clean text columns
        for col in text_columns:
            new_job_dict[col] = new_job_dict.get(col, 'unspecified')
            new_job_dict[col] = text_clean(new_job_dict[col])
        
        # Combine text
        combined_text = ' '.join(new_job_dict[col] for col in text_columns)
        
        # Transform features
        X_text_new = vectorizer.transform([combined_text])
        cat_df = pd.DataFrame({col: [new_job_dict.get(col, 'unspecified')] for col in categorical_columns})
        X_cat_new = encoder.transform(cat_df)
        X_new = hstack([X_text_new, X_cat_new])
        
        # Predict
        prob = model.predict_proba(X_new)[0][1]
        result = 'Fake' if prob >= threshold else 'Real'
        
        return result, round(prob * 100, 2)

    # Flask Route 
    @app.route('/predict', methods=['POST'])
    def predict():
        try:
            data = request.get_json()
            if not data:
                return jsonify({'error': 'No JSON data provided'}), 400
            
            threshold = data.get('threshold', 0.5)
            result, prob_percent = predict_job(data, threshold)
            
            return jsonify({
                "prediction": result,
                "fake_probability_percent": prob_percent,
                "message": "Prediction successful"
            })
        except Exception as e:
            return jsonify({"error": str(e)}), 500

    # Run Flask in Background 
    if __name__ == '__main__':
        print('🚀 Flask API is starting...')
        print('Test URL: http://127.0.0.1:5000/predict')
        app.run(debug=False, host='0.0.0.0', port=5000)