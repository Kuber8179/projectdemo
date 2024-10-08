---
license: mit
---

# Healthcare Chatbot

This project is an AI-driven healthcare chatbot designed to classify medications and provide detailed medical information based on user inputs. The chatbot is equipped with Natural Language Processing (NLP) and deep learning models, allowing it to handle various healthcare-related queries effectively.
## Features

- **Counseling & Greeting:** Engages users with conversational greetings and provides counseling based on user input.
- **Medication Classification:** Uses a machine learning model to classify medications based on several input factors.
- **Medical Information:** Responds to user queries with detailed medical information.
- **Interactive UI:** A user-friendly, dark-mode web interface that supports easy navigation and interaction.

## Technologies

- **Programming Languages:** Python, JavaScript, HTML, CSS
- **Frameworks & Libraries:** Flask, TensorFlow, scikit-learn, VGG16, - NLP libraries (e.g., NLTK, spaCy)
- **Frontend:** HTML, CSS, JavaScript
- **Backend:** Flask (Python)
- **Machine Learning Models:** Custom TensorFlow models, VGG16 for feature extraction, scikit-learn classifiers
## Models

Medication Classification Model
- **Input:** Age, Gender, Blood Type, Medical Condition, Test Results
- **Output:** Medication class
- **Model Type:** scikit-learn classifier trained on structured data.
Medicine Information Model
- **Input:** User medical query
- **Output:** Detailed medical advice
- **Model Type:** Pre-trained NLP model with custom fine-tuning.
Counselling Model
- **Input:** User medical/counsell query
- **Output:** Detailed medical/counsell advice
- **Model Type:** Pre-trained NLP model with custom fine-tuning.
## Usage

This project is designed to assist healthcare providers and patients by:

- Classifying medications based on specific user input.
- Providing personalized medical information to guide treatment decisions.
- Offering a user-friendly interface for both healthcare professionals and patients.

### Example Usage:
**Medication Classification:** Enter user details such as age, gender, blood type, medical condition, and test results to classify the most appropriate medication.

**Medical Query:** Input a medical question to receive detailed, AI-generated advice.

### WebApp

![App](https://huggingface.co/datasets/hassaanik/HealthCare_Bot_App/resolve/main/HealthCare%20Bot.gif)
