# fruit-veg-recognition
• Why this specific model despite all existing ones?
This is an AI-based real-time fruit and vegetable recognition system with dish recommendations, built using Streamlit.
It uses your webcam to continuously detect which fruit or vegetable is in front of it, and then suggests dishes you can prepare using that item.

• Why did i build this project?
Many people find it difficult to decide what to cook when they have fruits or vegetables at home. This project makes cooking more fun and creative by using AI to suggest ideas instantly.

• How does it work?
1️.Webcam Capture
The app uses your webcam to capture live video frames.

2️.AI Model Prediction
Each frame is resized and processed using a TensorFlow deep learning model you trained (fruit_veg_classifier.keras).
The model predicts which fruit or vegetable is present in the frame.

3️.Dish Suggestion
Once an item is recognized, the app looks up a list of predefined dishes related to that fruit or vegetable and displays them in real-time.

4️.Streamlit Interface
All of this happens inside an easy-to-use Streamlit web app, so you can see live video, predictions, and suggested dishes directly in your browser.

• Example scenario
You show a mango to your webcam.
The AI recognizes it as "mango."
The app suggests: Mango Lassi, Mango Ice Cream, Mango Salsa.

• Key Features
Real-time detection using your laptop or PC webcam
Automatic dish recommendations
Simple and interactive UI
Uses deep learning without requiring high-end resources

• Who can use it?
Home cooks and food enthusiasts who want quick dish ideas
Students learning about AI applications
Anyone who loves experimenting in the kitchen!

