from transformers import T5Tokenizer, TFT5ForConditionalGeneration, AutoTokenizer, TFAutoModelForSeq2SeqLM
import tensorflow as tf

# Load both models and tokenizers
model1 = TFT5ForConditionalGeneration.from_pretrained('models\\counsel_model\\saved_model')
tokenizer1 = T5Tokenizer.from_pretrained('models\\counsel_model\\tokenizer')

model2 = TFAutoModelForSeq2SeqLM.from_pretrained('models\\greeting_model\\saved_model')
tokenizer2 = AutoTokenizer.from_pretrained('models\\greeting_model\\saved_model')

def ensemble_generate(question):
    # Prepare the input for Model 1
    input_text1 = f"question: {question}"
    encoding1 = tokenizer1(
        input_text1,
        max_length=500,
        padding='max_length',
        truncation=True,
        return_tensors='tf'
    )

    input_ids1 = encoding1['input_ids']
    attention_mask1 = encoding1['attention_mask']

    # Generate output from Model 1
    outputs1 = model1.generate(
        input_ids=input_ids1,
        attention_mask=attention_mask1,
        max_length=512,
        num_beams=5,
        early_stopping=True,
        no_repeat_ngram_size=2,
        return_dict_in_generate=True,
        output_scores=True
    )

    generated_text1 = tokenizer1.decode(outputs1.sequences[0], skip_special_tokens=True)

    # Prepare the input for Model 2
    input_ids2 = tokenizer2.encode(question, return_tensors='tf')

    # Generate output from Model 2
    outputs2 = model2.generate(input_ids2, max_length=500, num_beams=4, early_stopping=True)
    generated_text2 = tokenizer2.decode(outputs2[0], skip_special_tokens=True)

    # Ensemble strategy: Simple concatenation of both responses
    final_response = f"Model 1 Response: {generated_text1}\nModel 2 Response: {generated_text2}"

    return final_response

# Test the ensemble with some questions
test_questions = [
    'What does it mean to have a mental illness?',
    'What are some of the warning signs of mental illness?',
    'What is the Information Technology syllabus?',
    'How are you? How is your day?',
    'Is anyone there?',
]

for question in test_questions:
    print(f"Question: {question}")
    print(f"Ensembled Answer: {ensemble_generate(question)}\n")
