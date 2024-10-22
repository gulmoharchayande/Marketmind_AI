# MarketMind AI

MarketMind AI is an innovative AI-powered marketing assistant that helps businesses streamline their marketing efforts through content generation, customer behavior prediction, and personalized campaign creation.

## Features

1. **AI Content Generation**: Create engaging, topic-specific content with customizable tones.
2. **Customer Behavior Prediction**: Utilize machine learning to predict customer behavior based on historical data.
3. **Personalized Campaign Creation**: Generate tailored marketing campaigns for specific target audiences and products.

## Installation

To run MarketMind AI locally, follow these steps:

1. Clone the repository:
   ```
   git clone https:https://github.com/gulmoharchayande/AI_marketing_agency
   cd marketmind-ai
   ```

2. Create a virtual environment:
   ```
   python -m venv venv
   ```

3. Activate the virtual environment:
   - On Windows:
     ```
     venv\Scripts\activate
     ```
   - On macOS and Linux:
     ```
     source venv/bin/activate
     ```

4. Install the required packages:
   ```
   pip install -r requirements.txt
   ```

5. Run the Streamlit app:
   ```
   streamlit run app.py
   ```

## Usage

### Content Generation
1. Select "Content Generation" from the sidebar.
2. Enter a topic and select the desired tone.
3. Click "Generate Content" to create AI-generated marketing content.

### Customer Behavior Prediction
1. Select "Customer Behavior Prediction" from the sidebar.
2. Upload a CSV file containing customer data.
   - Ensure the CSV has feature columns and a target column (last column).
3. Click "Predict Behavior" to see predictions and model accuracy.

### Personalized Campaign
1. Select "Personalized Campaign" from the sidebar.
2. Enter the target audience, product/service, and campaign goal.
3. Click "Generate Campaign" to create a personalized marketing strategy.

## Technologies Used

- Streamlit
- Pandas
- Scikit-learn
- Hugging Face Transformers
- PyTorch

## License

NA

## Contact

Gulmohar Chayande - [chayandegulmohar@outlook.com]

## Acknowledgments

- Thanks to the Streamlit team for their amazing framework.
- Hugging Face for providing powerful NLP models.
- The scikit-learn community for their robust machine learning tools.
