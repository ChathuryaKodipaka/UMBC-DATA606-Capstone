import streamlit as st
import tensorflow as tf
import pandas as pd
from PIL import Image
import numpy as np
import base64
from io import BytesIO

# Set Streamlit to wide layout and center the title
st.set_page_config(layout="wide")
st.markdown("<h1 style='text-align: center;'>Emotion-Based Music Recommendation</h1>", unsafe_allow_html=True)
st.markdown("<div style='text-align: center;'>Upload an image or capture one with your webcam, and we'll recommend songs based on your detected emotion!</div>", unsafe_allow_html=True)

# Load the trained model
model = tf.keras.models.load_model('/Users/chathurya/Desktop/Emotion_Detection/ResNet50V2_Model/ResNet50V2_Output/resnet50v2_model.keras')

# Define class labels and mood mapping
class_labels = {0: 'angry', 1: 'disgust', 2: 'fear', 3: 'happy', 4: 'neutral', 5: 'sad', 6: 'surprise'}
emotion_to_mood = {'happy': 'happy', 'neutral': 'calm', 'angry': 'sad', 'fear': 'sad', 'disgust': 'sad', 'surprise': 'energetic', 'sad': 'sad'}

# Load music data and convert mood column to lowercase
csv_path = '/Users/chathurya/Desktop/Emotion_Detection/Spotify_Music/data_moodS.csv'
data_mood = pd.read_csv(csv_path)
data_mood['mood'] = data_mood['mood'].str.lower()

# Initialize filtered_songs as None
filtered_songs = None

# Image source selection
st.write("## Choose an option to provide an image:")
option = st.selectbox("Select Image Source:", ("Upload an Image", "Capture with Webcam"))

# Layout for image upload or webcam capture
col1, col2 = st.columns([1, 1])

image = None
detected_emotion = None
with col1:
    if option == "Upload an Image":
        st.write("## Upload an image")
        uploaded_file = st.file_uploader("Drag and drop or browse a file", type=["jpg", "jpeg", "png"])
        if uploaded_file is not None:
            image = Image.open(uploaded_file)

    elif option == "Capture with Webcam":
        st.write("## Capture with Webcam")
        captured_image = st.camera_input("Capture an image to detect your mood...")
        if captured_image is not None:
            image = Image.open(captured_image)

if image is not None:
    # Display the image and detected emotion in the right column
    with col2:
        # Convert image to bytes and display without encoding issues
        buffered = BytesIO()
        image.save(buffered, format="JPEG")
        img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")

        st.markdown(
            f"""
            <div style='display: flex; justify-content: center; align-items: center; height: 100%; flex-direction: column;'>
                <img src="data:image/jpeg;base64,{img_str}" width="300" style="border-radius: 8px; margin-bottom: 20px;" alt="Uploaded Image"/>
            </div>
            """,
            unsafe_allow_html=True
        )

        # Preprocess and predict emotion
        image = image.convert("RGB").resize((224, 224))
        image_array = np.array(image) / 255.0
        image_array = np.expand_dims(image_array, axis=0)
        prediction = model.predict(image_array)
        predicted_class = np.argmax(prediction, axis=1)[0]
        detected_emotion = class_labels[predicted_class]

        # Display detected emotion right below the image
        st.markdown(f"<h3 style='text-align: center; margin-top: 10px;'>Detected Emotion: {detected_emotion}</h3>", unsafe_allow_html=True)

        # Map the detected emotion to a mood for song recommendation
        recommended_mood = emotion_to_mood.get(detected_emotion, 'calm').lower()
        
        # Filter songs based on the recommended mood and randomly sample 5 songs
        filtered_songs = data_mood[data_mood['mood'] == recommended_mood].sample(n=5, random_state=None) if data_mood is not None else None

# Display Recommended Songs in a styled HTML table format
st.markdown("<div style='text-align: center; margin-top: 50px;'><h3>Recommended Songs:</h3></div>", unsafe_allow_html=True)

if filtered_songs is not None and not filtered_songs.empty:
    st.markdown("""
    <style>
        table {
            width: 80%;
            margin: 0 auto;
            border-collapse: collapse;
            font-family: Arial, sans-serif;
        }
        th, td {
            padding: 12px;
            text-align: center;
            border: 1px solid #ddd;
        }
        th {
            background-color: #343a40;
            color: white;
            font-weight: bold;
        }
        tr:nth-child(even) {
            background-color: #f2f2f2;
        }
    </style>
    """, unsafe_allow_html=True)

    # Generate HTML table with song data
    table_html = "<table><tr><th>Song</th><th>Album</th><th>Energy Level</th></tr>"
    for index, row in filtered_songs.iterrows():
        table_html += f"<tr><td>{row['name']} by {row['artist']}</td><td>{row['album']}</td><td>{round(row['energy'], 2)}</td></tr>"
    table_html += "</table>"

    st.markdown(table_html, unsafe_allow_html=True)