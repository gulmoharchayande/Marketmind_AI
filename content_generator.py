from transformers import pipeline

# Initialize the text generation pipeline
generator = pipeline('text-generation', model='gpt2')


def generate_content(topic, tone):
    prompt = f"Write a {tone.lower()} article about {topic}:\n\n"

    # Generate text
    result = generator(prompt, max_length=300, num_return_sequences=1)

    # Extract the generated text
    generated_text = result[0]['generated_text']

    # Remove the prompt from the generated text
    final_text = generated_text.replace(prompt, "")

    return final_text.strip()
