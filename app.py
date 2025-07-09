import streamlit as st
import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import img_to_array

# Load trained model
model = tf.keras.models.load_model("fruit_veg_classifier.keras")

# Define dishes based on fruit/vegetable
fruit_dishes = {
    "apple": ["Apple Juice", "Apple Pie", "Apple Salad"],
    "banana": ["Banana Milkshake", "Banana Pancakes", "Banana Smoothie"],
    "grape": ["Grape Juice", "Fruit Salad", "Grape Sorbet"],
    "mango": ["Mango Lassi", "Mango Ice Cream", "Mango Salsa"],
    "strawberry": ["Strawberry Smoothie", "Strawberry Shortcake", "Strawberry Jam"],
    "orange": ["Orange Juice", "Orange Cake", "Citrus Salad"],
    "pineapple": ["Pineapple Juice", "Pineapple Pizza", "Pineapple Fried Rice"],
    "carrot": ["Carrot Halwa", "Carrot Soup", "Carrot Salad"],
    "potato": ["Mashed Potatoes", "French Fries", "Potato Curry"],
}

# Function to predict fruit or vegetable
def predict_fruit(image):
    img_array = img_to_array(image) / 255.0  # Normalize image
    img_array = np.expand_dims(img_array, axis=0)  # Expand dimensions

    predictions = model.predict(img_array)
    predicted_class_index = np.argmax(predictions)  # Get class index

    class_names = list(fruit_dishes.keys())  # Load class names
    return class_names[predicted_class_index] if predicted_class_index < len(class_names) else "Unknown"

# Function to suggest dishes
def suggest_dishes(fruit_name):
    return fruit_dishes.get(fruit_name.lower(), ["No dish suggestions available."])

# Streamlit UI
st.title("🍎 AI-based Fruit & Vegetable Recognition 🍌")
st.write("Real-time webcam-based fruit & vegetable detection with dish recommendations!")

# Webcam live streaming
video_capture = cv2.VideoCapture(0)  # Open webcam

stframe = st.empty()  # Placeholder for live video feed

while True:
    ret, frame = video_capture.read()
    if not ret:
        st.write("Error: Cannot access webcam.")
        break

    # Convert frame to RGB (for Streamlit)
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    frame_resized = cv2.resize(frame, (224, 224))  # Resize for model

    # Predict fruit or vegetable
    detected_fruit = predict_fruit(frame_resized)
    recommended_dishes = suggest_dishes(detected_fruit)

    # Display live video feed
    stframe.image(frame, caption=f"Detected: {detected_fruit.capitalize()}", channels="RGB", use_column_width=True)

    # Display dish recommendations
    st.subheader(f"🍽 Recommended Dishes for {detected_fruit.capitalize()}:")
    for dish in recommended_dishes:
        st.write(f"- {dish}")

    # Stop the stream when user clicks "Stop"
    if st.button("Stop Streaming"):
        break

# Release webcam
video_capture.release()
