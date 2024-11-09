from flask import Flask, render_template, Response, redirect, url_for  # Import redirect and url_for
import cv2
import numpy as np
from keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
import time

app = Flask(__name__)
@app.route('/')
def index():
    return render_template('index.html')  # The HTML with the button

@app.route('/start_model_process')
def start_model_process():
    # Start the model process here. For example, capturing frames or any model-related process.
    # You can replace this with your actual model code.
    result = run_model_process()
    return render_template('result.html', result=result)

def run_model_process():
    # Simulate running model process (you can replace this with actual code)
    # For example, capturing an image, processing, etc.
    time.sleep(2)  # Simulate some delay for model processing
    return "Real"  # Or return "Fake" depending on model output

if __name__ == '__main__':
    app.run(debug=True)


# Load face liveness model and cascade classifier
model = load_model(r"C:\Users\Mayank Meghwal\Desktop\FaceDetection\Face-Liveness-Detection\livenessdetect\models\anandfinal.hdf5")
faceCascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/result')
def result():
    # This is just a placeholder. You would pass the actual results here.
    return render_template('result.html', confidence=75, result="Real", image_url="static/captured_frame.jpg")

def detect_liveness(frame):
    global liveness_result, liveness_confidence
    # Convert frame to grayscale and detect face
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = faceCascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    for (x, y, w, h) in faces:
        face_region = frame[y:y+h, x:x+w]
        face_region = cv2.resize(face_region, (128, 128))
        face_region = img_to_array(face_region) / 255.0
        face_region = np.expand_dims(face_region, axis=0)

        # Predict if face is real or fake
        (real, fake) = model.predict(face_region)[0]
        liveness_confidence = max(real, fake) * 100
        liveness_result = "Real" if real > fake else "Fake"
        
        # Draw bounding box and label
        color = (0, 255, 0) if liveness_result == "Real" else (0, 0, 255)
        label_text = f"{liveness_result} ({liveness_confidence:.2f}%)"
        cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
        cv2.putText(frame, label_text, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

        # Save the captured frame
        cv2.imwrite("static/captured_frame.jpg", frame)

    return frame

def generate_frames():
    camera = cv2.VideoCapture(0)
    start_time = time.time()
    
    # Capture video for 5 seconds
    while True:
        success, frame = camera.read()
        if not success:
            break

        # Resize the frame to 500x500 for liveness detection
        frame = cv2.resize(frame, (500, 500))

        # Perform liveness detection
        frame = detect_liveness(frame)

        # Encode frame to JPEG for streaming
        ret, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()

        yield (b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

        # Stop streaming after 5 seconds and redirect to result page
        if time.time() - start_time > 5:
            break

    camera.release()
    # After video feed ends, redirect to result page
    return redirect(url_for('result'))

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == "__main__":
    app.run(debug=True)
