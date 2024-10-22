import streamlit as st
import pandas as pd
from content_generator import generate_content
from customer_predictor import predict_customer_behavior
from personalization_engine import generate_personalized_campaign


def main():
    st.set_page_config(page_title="MarketMind AI", page_icon="🚀", layout="wide")
    st.title("MarketMind AI")

    menu = ["Home", "Content Generation", "Customer Behavior Prediction", "Personalized Campaign"]
    choice = st.sidebar.selectbox("Select a Service", menu)

    if choice == "Home":
        display_home()
    elif choice == "Content Generation":
        display_content_generation()
    elif choice == "Customer Behavior Prediction":
        display_customer_behavior_prediction()
    elif choice == "Personalized Campaign":
        display_personalized_campaign()


def display_home():
    st.subheader("Welcome to AI Marketing Agency")
    st.write(
        "We use advanced AI to revolutionize your marketing strategies. Choose a service from the sidebar to get started.")
    st.write("Our services include:")
    st.write("1. AI-powered content generation")
    st.write("2. Customer behavior prediction using machine learning")
    st.write("3. Personalized marketing campaign creation")


def display_content_generation():
    st.subheader("AI Content Generation")
    topic = st.text_input("Enter the topic for content generation:")
    tone = st.selectbox("Select the tone of the content:", ["Professional", "Casual", "Humorous", "Inspirational"])
    if st.button("Generate Content"):
        if topic:
            with st.spinner("Generating content..."):
                try:
                    generated_content = generate_content(topic, tone)
                    st.subheader("Generated Content:")
                    st.write(generated_content)
                except Exception as e:
                    st.error(f"An error occurred during content generation: {str(e)}")
        else:
            st.warning("Please enter a topic.")


def display_customer_behavior_prediction():
    st.subheader("Customer Behavior Prediction")
    uploaded_file = st.file_uploader("Upload customer data CSV", type="csv")
    if uploaded_file is not None:
        try:
            data = pd.read_csv(uploaded_file)
            st.write("Preview of uploaded data:")
            st.write(data.head())
            if st.button("Predict Behavior"):
                with st.spinner("Predicting behavior..."):
                    predictions, accuracy = predict_customer_behavior(data)
                    st.subheader("Customer Behavior Predictions:")
                    st.write(predictions)
                    st.write(f"Model Accuracy: {accuracy:.2f}")
        except Exception as e:
            st.error(f"An error occurred while processing the file: {str(e)}")
    else:
        st.info("Please upload a CSV file with customer data.")


def display_personalized_campaign():
    st.subheader("Personalized Marketing Campaign")
    target_audience = st.text_input("Describe your target audience:")
    product = st.text_input("Enter your product or service:")
    campaign_goal = st.selectbox("Select campaign goal:", ["Brand Awareness", "Lead Generation", "Sales Conversion"])
    if st.button("Generate Campaign"):
        if target_audience and product:
            with st.spinner("Generating campaign..."):
                try:
                    campaign = generate_personalized_campaign(target_audience, product, campaign_goal)
                    st.subheader("Personalized Marketing Campaign:")
                    st.write(campaign)
                except Exception as e:
                    st.error(f"An error occurred during campaign generation: {str(e)}")
        else:
            st.warning("Please enter both target audience and product information.")


if __name__ == "__main__":
    main()
