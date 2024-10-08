import joblib
import pandas as pd

def predict_medication(new_data):
    # Load the model, encoders, and scaler
    knn = joblib.load('models\\medication_classification_model\\knn_model.pkl')
    label_encoders = joblib.load('models\\medication_classification_model\\label_encoders.pkl')
    age_scaler = joblib.load('models\\medication_classification_model\\age_scaler.pkl')
    medication_encoder = joblib.load('models\\medication_classification_model\\medication_encoder.pkl')

    # Encode the new data using the saved label encoders
    for column in ['Gender', 'Blood Type', 'Medical Condition', 'Test Results']:
        new_data[column] = label_encoders[column].transform(new_data[column])

    # Normalize the 'Age' column in the new data
    new_data['Age'] = age_scaler.transform(new_data[['Age']])

    # Make predictions
    predictions = knn.predict(new_data)

    # Decode the predictions back to the original medication names
    predicted_medications = medication_encoder.inverse_transform(predictions)

    return predicted_medications
