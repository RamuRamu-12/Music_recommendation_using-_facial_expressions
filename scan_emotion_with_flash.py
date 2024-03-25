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
df.rename(
    columns={'lastfm_url': 'Link', 'track': 'Name', 'number_of_emotion_tags': 'Emotional', 'valence_tags': 'Pleasant'},
    inplace=True)

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

# Configure OpenCV
cv2.ocl.setUseOpenCL(False)
cap = cv2.VideoCapture(0)

# Initialize emotion_list
emotion_list = []

# Streamlit UI
st.markdown("<h2 style='text-align: center;color:white;'><b>Emotion Based Music Recommendation</b></h2>",
            unsafe_allow_html=True)
st.markdown(
    "<h5 style='text-align: center;color:grey;'><b>Click on the name of the recommended song to reach the website</b></h5>",
    unsafe_allow_html=True)

def calculate_accuracy(detected_emotion, recommended_songs):
    total_recommendations = len(recommended_songs)
    correct_predictions = sum(recommended_songs['Emotional'] == detected_emotion)
    accuracy = (correct_predictions / total_recommendations) * 100
    return accuracy

# Placeholder for buttons
col1, col2, col3 = st.columns(3)
with col1:
    pass

with col2:
    if st.button("SCAN EMOTION (Click Here)"):
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            face_cascade = cv2.CascadeClassifier(
                'C:/Users/rammohan/PycharmProjects/pythonProject1/Data_Science/final_year/haarcascade_frontalface_default.xml')
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)
            for (x, y, w, h) in faces:
                cv2.rectangle(frame, (x, y - 50), (x + w, y + h + 10), (255, 0, 0), 2)
                roi_gray = gray[y:y + h, x:x + w]
                cropped_img = np.expand_dims(np.expand_dims(cv2.resize(roi_gray, (48, 48)), -1), 0)
                prediction = model.predict(cropped_img)
                max_index = int(np.argmax(prediction))
                if len(emotion_list) >= 30:
                    emotion_list.pop(0)  # Remove the oldest emotion if the list exceeds 30
                emotion_list.append(max_index)
                cv2.putText(frame, emotion_dict[max_index], (x + 20, y - 60), cv2.FONT_HERSHEY_SIMPLEX, 1,
                            (255, 255, 255), 2)
                cv2.imshow('Video', cv2.resize(frame, (1000, 700), interpolation=cv2.INTER_CUBIC))

                # Check for low light conditions and activate flashlight
                mean_intensity = np.mean(roi_gray)
                if mean_intensity < 50:  # Adjust this threshold as needed
                    # Activate flashlight
                    os.system("xset dpms force on")  # Command to turn on the screen
                else:
                    # Deactivate flashlight
                    os.system("xset dpms force off")  # Command to turn off the screen
            if cv2.waitKey(1) & 0xFF == ord('x'):
                break

        # Close OpenCV window
        cap.release()
        cv2.destroyAllWindows()

        # Display the detected emotions
        st.write("Detected Emotions:", [emotion_dict[i] for i in emotion_list])

# Placeholder for buttons
with col3:
    pass

# Process the last frame to recommend songs based on detected emotions
if len(emotion_list) > 0:
    detected_emotion = max(set(emotion_list), key=emotion_list.count)
    st.write("Detected Emotion for Recommendation:", emotion_dict[detected_emotion])

    # Randomly select songs from the dataset
    random_songs = df_1.sample(n=15)

    # Display recommendations based on the detected emotions
    st.write("")
    st.markdown(
        "<h5 style='text-align:center;color:grey;'><b>Recommended songs with artist names based on the detected emotion</b></h5>",
        unsafe_allow_html=True)
    st.write("-------------------------------------------------------------")

    accuracy = calculate_accuracy(detected_emotion, random_songs)
    print(f"Accuracy of song recommendation: {accuracy:.2f}%")

    # Display recommended songs based on the detected emotion
    try:
        for index, row in random_songs.iterrows():
            st.markdown(f"""<h4 style='text-align:center;'><a href={row['Link']}>{row['Name']}</a></h4>""",
                        unsafe_allow_html=True)
            st.markdown(f"<h5 style='text-align:center;color:gray;'><i>{row['artist']}</i></h5>",
                        unsafe_allow_html=True)
            st.write("------------------------------------------------------")
    except Exception as e:
        print("Error displaying recommendations:", e)
else:
    st.write("No emotion detected. Please click on 'SCAN EMOTION' button.")  # Display message if no emotion detected
