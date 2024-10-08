from transformers import T5Tokenizer, TFT5ForConditionalGeneration
import tensorflow as tf

# Load the model and tokenizer
model = TFT5ForConditionalGeneration.from_pretrained('models\\medication_info_model\\saved_model')
tokenizer = T5Tokenizer.from_pretrained('models\\medication_info_model\\tokenizer')

def generate_answer(question):
    input_text = f"question: {question}"
    encoding = tokenizer(
        input_text,
        max_length=1024,
        padding='max_length',
        truncation=True,
        return_tensors='tf'
    )

    input_ids = encoding['input_ids']
    attention_mask = encoding['attention_mask']

    generated_text = ""
    max_length = 1024
    current_input_ids = input_ids

    while True:
        outputs = model.generate(
            input_ids=current_input_ids,
            attention_mask=attention_mask,
            max_length=max_length,
            num_beams=5,
            early_stopping=True,
            no_repeat_ngram_size=2,
            return_dict_in_generate=True,
            output_scores=True
        )

        text = tokenizer.decode(outputs.sequences[0], skip_special_tokens=True)
        generated_text += text

        if len(text.split()) < max_length:
            break

        current_input_ids = tokenizer.encode(text, return_tensors='tf')
        attention_mask = tf.ones_like(current_input_ids)

    return generated_text
