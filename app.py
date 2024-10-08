from flask import Flask, render_template, request, jsonify, session
import tensorflow as tf
from transformers import T5Tokenizer, TFT5ForConditionalGeneration
import joblib
import pandas as pd
import datetime

app = Flask(__name__)
app.secret_key = 'hassaanik'  # Necessary for session management


# Load models and tokenizers
counseling_greeting_model = TFT5ForConditionalGeneration.from_pretrained('./models/counseling_greeting_model/saved_model')
counseling_greeting_tokenizer = T5Tokenizer.from_pretrained('./models/counseling_greeting_model/tokenizer')

med_info_model = TFT5ForConditionalGeneration.from_pretrained('./models/medication_info_model/saved_model')
med_info_tokenizer = T5Tokenizer.from_pretrained('./models/medication_info_model/tokenizer')

knn_model = joblib.load('./models/medication_classification_model/knn_model.pkl')
label_encoders = joblib.load('./models/medication_classification_model/label_encoders.pkl')
age_scaler = joblib.load('./models/medication_classification_model/age_scaler.pkl')
medication_encoder = joblib.load('./models/medication_classification_model/medication_encoder.pkl')


# Existing model loading code...

@app.route('/')
def index():
    session.clear()  # Clear session when accessing the homepage
    return render_template('index.html')

@app.route('/reset_chat', methods=['POST'])
def reset_chat():
    session.clear()
    return jsonify({'status': 'Chat reset'})

def generate_response(model, tokenizer, input_text, session_key):
    # Prepare input for model
    encoding = tokenizer(input_text, max_length=500, padding='max_length', truncation=True, return_tensors='tf')
    input_ids = encoding['input_ids']
    attention_mask = encoding['attention_mask']
    # Generate response
    outputs = model.generate(input_ids=input_ids, attention_mask=attention_mask, max_length=512, num_beams=5, early_stopping=True)
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    # Store in session
    if session_key not in session:
        session[session_key] = []
    session[session_key].append({'user': input_text, 'bot': response})
    return response

@app.route('/counseling_greeting', methods=['POST'])
def counseling_greeting():
    data = request.get_json()
    prompt = data['prompt']
    response = generate_response(counseling_greeting_model, counseling_greeting_tokenizer, f"question: {prompt}", 'counseling_greeting')
    return jsonify({'response': response, 'conversation': session['counseling_greeting']})

@app.route('/medication_info', methods=['POST'])
def medication_info():
    data = request.get_json()
    question = data['question']
    response = generate_response(med_info_model, med_info_tokenizer, f"question: {question}", 'medication_info')
    return jsonify({'response': response, 'conversation': session['medication_info']})

@app.route('/classify_medication', methods=['POST'])
def classify_medication():
    data = pd.DataFrame([request.get_json()])
    for column in ['Gender', 'Blood Type', 'Medical Condition', 'Test Results']:
        data[column] = label_encoders[column].transform(data[column])
    data['Age'] = age_scaler.transform(data[['Age']])
    predictions = knn_model.predict(data)
    predicted_medications = medication_encoder.inverse_transform(predictions)
    if 'classify_medication' not in session:
        session['classify_medication'] = []
    session['classify_medication'].append({'user': data.to_dict(), 'bot': predicted_medications[0]})
    return jsonify({'medication': predicted_medications[0], 'conversation': session['classify_medication']})


if __name__ == '__main__':
    app.run(debug=True)
