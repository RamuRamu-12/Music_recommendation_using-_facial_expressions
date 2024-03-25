import numpy as np
import streamlit as st
import pandas as pd
import cv2
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
import os

# Set environment variable to turn off oneDNN custom operations
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

# Load the dataset
df = pd.read_csv("C:/Users/rammohan/PycharmProjects/pythonProject1/Data_Science/final_year/muse_v3.csv")

# Rename columns for clarity
df.rename(columns={'lastfm_url': 'Link', 'track': 'Name', 'number_of_emotion_tags': 'Emotional', 'valence_tags': 'Pleasant'}, inplace=True)

# Select relevant columns
df_1 = df[["Name", "Emotional", "Pleasant", "Link", "artist"]]

# Load the pre-trained emotion detection model
model = Sequential()
model.add(Conv2D(32, kernel_size=(3, 3), activation="relu", input_shape=(48, 48, 1)))
model.add(Conv2D(64, kernel_size=(3, 3), activation="relu"))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))
model.add(Conv2D(128, kernel_size=(3, 3), activation="relu"))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(128, kernel_size=(3, 3), activation="relu"))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))
model.add(Flatten())
model.add(Dense(1024, activation="relu"))
model.add(Dropout(0.5))
model.add(Dense(7, activation="softmax"))
model.load_weights('C:/Users/rammohan/PycharmProjects/pythonProject1/Data_Science/final_year/model.h5')

# Dictionary to map emotion indices to labels
emotion_dict = {0: "Angry", 1: "Disgusted", 2: "Fearful", 3: "Happy", 4: "Neutral", 5: "Sad", 6: "Surprised"}

# Function to preprocess emotions list
def preprocess_emotions(emotion_list):
    # Preprocessing logic to sort emotions based on count
    return sorted(emotion_list, key=lambda x: emotion_list.count(x), reverse=True)

# Function to detect low light conditions
def is_low_light(image, threshold=50):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    mean_intensity = np.mean(gray)
    return mean_intensity < threshold

# Streamlit UI
st.markdown("<h2 style='text-align: center;color:white;'><b>Emotion Based Music Recommendation</b></h2>", unsafe_allow_html=True)
st.markdown("<h5 style='text-align: center;color:grey;'><b>Click on the name of the recommended song to reach the website</b></h5>", unsafe_allow_html=True)

# File uploader for image
uploaded_image = st.file_uploader("Upload an image", type=["jpg", "png", "jpeg"])

if uploaded_image is not None:
    # Read the uploaded image
    file_bytes = np.asarray(bytearray(uploaded_image.read()), dtype=np.uint8)
    frame = cv2.imdecode(file_bytes, 1)

    # Check for low light condition
    if is_low_light(frame):
        st.write("Low light detected in the uploaded image. Please try again with a better-lit image.")
    else:
        # Display uploaded image
        st.image(frame, channels="BGR", caption="Uploaded Picture")

        # Detect faces in the image
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        face_cascade = cv2.CascadeClassifier('C:/Users/rammohan/PycharmProjects/pythonProject1/Data_Science/final_year/haarcascade_frontalface_default.xml')
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)

        if len(faces) > 0:
            # Take only the first face detected
            (x, y, w, h) = faces[0]

            # Crop and resize the face region
            roi_gray = gray[y:y+h, x:x+w]
            cropped_img = cv2.resize(roi_gray, (48, 48))

            # Preprocess the image for prediction
            cropped_img = np.expand_dims(np.expand_dims(cropped_img, -1), 0)

            # Predict emotion
            prediction = model.predict(cropped_img)
            max_index = int(np.argmax(prediction))
            detected_emotion = emotion_dict[max_index]

            # Filter dataframe based on detected emotion
            recommended_songs = df_1[df_1['Emotional'] == max_index].head(15)  # Assuming you want to display top 10 recommendations

            # Display detected emotion
            st.write(f"Detected Emotion: {detected_emotion}")

            # Display recommendations based on the detected emotion
            st.write("")
            st.markdown("<h5 style='text-align:center;color:grey;'><b>Recommended songs with artist names based on the detected emotion</b></h5>", unsafe_allow_html=True)
            st.write("-------------------------------------------------------------")

            # Display recommended songs based on the detected emotion
            try:
                for index, row in recommended_songs.iterrows():
                    st.markdown(f"""<h4 style='text-align:center;'><a href={row['Link']}>{row['Name']}</a></h4>""", unsafe_allow_html=True)
                    st.markdown(f"<h5 style='text-align:center;color:gray;'><i>{row['artist']}</i></h5>", unsafe_allow_html=True)
                    st.write("------------------------------------------------------")
            except Exception as e:
                print("Error displaying recommendations:", e)
        else:
            st.write("No face detected in the uploaded image.")
