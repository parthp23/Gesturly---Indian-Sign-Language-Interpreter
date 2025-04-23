from flask import Flask, request, jsonify, render_template
import tensorflow as tf
import numpy as np
import mediapipe as mp
import cv2
import os
import tempfile
from gtts import gTTS
from flask_cors import CORS
import base64
import pandas as pd
import string
import itertools
import time
import copy

app = Flask(__name__, template_folder='templates')
CORS(app) # Enable CORS for all routes

# Load model and words
model = tf.keras.models.load_model("model.h5")

# Load words
with open("customwords.txt", "r") as f:
    custom_words = [line.strip().upper() for line in f if line.strip()]

# Alphabet
alphabet = ['1','2','3','4','5','6','7','8','9'] + list(string.ascii_uppercase)

# MediaPipe
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

# Initialize MediaPipe hands model
hands = mp_hands.Hands(model_complexity=0, max_num_hands=2,
                        min_detection_confidence=0.5, min_tracking_confidence=0.5)

# Globals (These will be managed server-side)
predicted_letters = []
last_gesture = None
hold_start_time = None
hold_duration = 0.7
cooldown_duration = 1.5
last_added_time = 0

# Helper functions
def calc_landmark_list(image, landmarks):
    image_width, image_height = image.shape[1], image.shape[0]

    landmark_point = []

    # Keypoint
    for _, landmark in enumerate(landmarks.landmark):
        landmark_x = min(int(landmark.x * image_width), image_width - 1)
        landmark_y = min(int(landmark.y * image_height), image_height - 1)
        # landmark_z = landmark.z

        landmark_point.append([landmark_x, landmark_y])

    return landmark_point

def pre_process_landmark(landmark_list):
    temp_landmark_list = copy.deepcopy(landmark_list)

    # Convert to relative coordinates
    base_x, base_y = 0, 0
    for index, landmark_point in enumerate(temp_landmark_list):
        if index == 0:
            base_x, base_y = landmark_point[0], landmark_point[1]

        temp_landmark_list[index][0] = temp_landmark_list[index][0] - base_x
        temp_landmark_list[index][1] = temp_landmark_list[index][1] - base_y

    # Convert to a one-dimensional list
    temp_landmark_list = list(
        itertools.chain.from_iterable(temp_landmark_list))

    # Normalization
    max_value = max(list(map(abs, temp_landmark_list)))

    def normalize_(n):
        return n / max_value

    temp_landmark_list = list(map(normalize_, temp_landmark_list))

    return temp_landmark_list

@app.route('/predict', methods=['POST'])
def predict():
    global last_gesture, hold_start_time, last_added_time, predicted_letters

    # Get the image data from the request
    data = request.get_json()
    img_data = data['image']
    # Decode base64 encoded image
    img_bytes = base64.b64decode(img_data.split(',')[1])
    npimg = np.frombuffer(img_bytes, np.uint8)
    img = cv2.imdecode(npimg, cv2.IMREAD_COLOR)

    # Flip the image horizontally for a later selfie-view display.
    img = cv2.flip(img, 1)

    # To improve performance, optionally mark the image as not writeable to pass by reference.
    img.flags.writeable = False
    image = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = hands.process(image)

    # Draw the hand annotations on the image.
    img.flags.writeable = True
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    debug_image = copy.deepcopy(img)
    label = None

    try:
        if results.multi_hand_landmarks:
            for hand_landmarks, handedness in zip(results.multi_hand_landmarks,results.multi_handedness):
                # Draw landmarks and hand connections on ROI
                mp_drawing.draw_landmarks(
                    img,
                    hand_landmarks,
                    mp_hands.HAND_CONNECTIONS,
                    mp_drawing_styles.get_default_hand_landmarks_style(),
                    mp_drawing_styles.get_default_hand_connections_style())

                landmark_list = calc_landmark_list(debug_image, hand_landmarks)
                pre_processed_landmark_list = pre_process_landmark(landmark_list)
                df = pd.DataFrame([pre_processed_landmark_list])

                # predict the sign language
                predictions = model.predict(df, verbose=0)
                # get the predicted class for each sample
                predicted_classes = np.argmax(predictions, axis=1)
                label = alphabet[predicted_classes[0]]

                current_time = time.time()
                if label == last_gesture:
                    if hold_start_time is None:
                        hold_start_time = current_time
                    elif current_time - hold_start_time >= hold_duration:
                        if current_time - last_added_time >= cooldown_duration:
                            predicted_letters.append(label)
                            last_added_time = current_time
                else:
                    last_gesture = label
                    hold_start_time = None
    except Exception as e:
        print(f"Error during processing: {e}")
        label = "Error"

    # Convert the image with landmarks back to base64 for display
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    _, img_encoded = cv2.imencode('.jpg', img)
    img_base64 = base64.b64encode(img_encoded).decode('utf-8')

    # Prepare the response
    result = {
        'image': img_base64,
        'predicted_text': ''.join(predicted_letters),
        'last_letter': label if label else "",
        'suggestions': [w for w in custom_words if w.startswith(''.join(predicted_letters).split()[-1].upper())][:3] if predicted_letters else []
    }
    return jsonify(result)

@app.route('/clear', methods=['POST'])
def clear():
    global predicted_letters, last_gesture, hold_start_time, last_added_time
    predicted_letters = []
    last_gesture = None
    hold_start_time = None
    last_added_time = 0
    return jsonify({'predicted_text': '', 'last_letter': ''})

@app.route('/undo', methods=['POST'])
def undo():
    global predicted_letters
    if predicted_letters:
        predicted_letters.pop()
    label = predicted_letters[-1] if predicted_letters else ""
    return jsonify({'predicted_text': ''.join(predicted_letters), 'last_letter': label})

@app.route('/update-text', methods=['POST'])
def update_text():
    global predicted_letters
    
    data = request.get_json()
    new_text = data.get('text', '')
    
    # Update the predicted_letters list with the new text
    # Converting from string to a list of characters
    predicted_letters = list(new_text)
    
    return jsonify({'status': 'success'})

@app.route('/speak', methods=['POST'])
def speak_text():
    data = request.get_json()
    text = data.get('text', '')
    
    if not text:
        return jsonify({'status': 'error', 'message': 'No text provided'})
    
    try:
        # Create a temporary file for the audio
        temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.mp3')
        temp_file.close()
        
        # Use gTTS to convert text to speech and save as MP3
        tts = gTTS(text=text, lang='en')
        tts.save(temp_file.name)
        
        # Read the audio file
        with open(temp_file.name, 'rb') as f:
            audio_data = f.read()
        
        # Clean up the temporary file
        os.unlink(temp_file.name)
        
        # Return the audio data as base64
        audio_base64 = base64.b64encode(audio_data).decode('utf-8')
        return jsonify({
            'status': 'success', 
            'audio': audio_base64
        })
        
    except Exception as e:
        print(f"Error in text-to-speech: {e}")
        return jsonify({
            'status': 'error',
            'message': str(e)
        })

@app.route('/')
def index():
    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=False, host='0.0.0.0', port=int(os.environ.get('PORT', 5000)))
