from transformers import pipeline

# Initialize the text generation pipeline
generator = pipeline('text-generation', model='gpt2')


def generate_personalized_campaign(target_audience, product, campaign_goal):
    prompt = f"Create a marketing campaign for {product} targeting {target_audience}. The campaign goal is {campaign_goal}. Include key messages, suggested channels, and creative ideas:\n\n"

    # Generate text
    result = generator(prompt, max_length=500, num_return_sequences=1)

    # Extract the generated text
    generated_text = result[0]['generated_text']

    # Remove the prompt from the generated text
    final_text = generated_text.replace(prompt, "")

    return final_text.strip()
