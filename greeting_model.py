import tensorflow as tf
from transformers import AutoTokenizer, TFAutoModelForSeq2SeqLM

# Load the fine-tuned model and tokenizer
model = TFAutoModelForSeq2SeqLM.from_pretrained('models\\greeting_model\\saved_model')
tokenizer = AutoTokenizer.from_pretrained('models\\greeting_model\\saved_model')

def generate_response(input_text, max_length=500):
    # Tokenize the input text
    input_ids = tokenizer.encode(input_text, return_tensors='tf')
    
    # Generate the response from the model
    outputs = model.generate(input_ids, max_length=max_length, num_beams=4, early_stopping=True)
    
    # Decode the generated tokens back to text
    decoded_response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    return decoded_response
