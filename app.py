import os
import cv2
import numpy as np
from flask import Flask, render_template, request, jsonify, Response
import tensorflow as tf

app = Flask(__name__)

# Directory setup
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
UPLOAD_FOLDER = os.path.join(BASE_DIR, 'static', 'uploads')
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Model & Face Cascade Loading
MODEL_PATH = os.path.join(BASE_DIR, 'face_mask_model.keras')
model = None

def get_model():
    global model
    if model is None:
        if os.path.exists(MODEL_PATH):
            model = tf.keras.models.load_model(MODEL_PATH)
            print("Keras model loaded successfully.")
        else:
            print("Model file face_mask_model.keras not found yet.")
    return model

# Safely load face cascade if available
face_cascade = None
if hasattr(cv2, 'CascadeClassifier') and hasattr(cv2, 'data'):
    cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
    if os.path.exists(cascade_path):
        face_cascade = cv2.CascadeClassifier(cascade_path)

CLASS_LABELS = {0: 'With Mask', 1: 'Without Mask'}

def preprocess_face(face_img):
    face_img = cv2.resize(face_img, (128, 128))
    face_img = cv2.cvtColor(face_img, cv2.COLOR_BGR2RGB)
    face_img = face_img.astype('float32') / 255.0
    return np.expand_dims(face_img, axis=0)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict_image():
    net = get_model()
    if net is None:
        return jsonify({'error': 'Model is still training or not found.'}), 400
        
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400
        
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400
        
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
    file.save(filepath)
    
    # Read image using OpenCV
    img = cv2.imread(filepath)
    if img is None:
        return jsonify({'error': 'Invalid image format'}), 400
        
    processed = preprocess_face(img)
    predictions = net.predict(processed)[0]
    
    pred_idx = np.argmax(predictions)
    confidence = float(predictions[pred_idx]) * 100
    label = CLASS_LABELS[pred_idx]
    
    # Draw prediction bounding box on image
    h, w, _ = img.shape
    color = (0, 200, 0) if pred_idx == 0 else (0, 0, 220)
    cv2.rectangle(img, (10, 10), (w - 10, h - 10), color, 4)
    text = f"{label}: {confidence:.1f}%"
    cv2.putText(img, text, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2, cv2.LINE_AA)
    
    output_filename = 'pred_' + file.filename
    output_path = os.path.join(app.config['UPLOAD_FOLDER'], output_filename)
    cv2.imwrite(output_path, img)
    
    return jsonify({
        'label': label,
        'confidence': round(confidence, 2),
        'image_url': f'/static/uploads/{output_filename}',
        'mask_status': 'success' if pred_idx == 0 else 'warning'
    })

def generate_frames():
    net = get_model()
    cap = cv2.VideoCapture(0) # Open default webcam
    
    if not cap.isOpened():
        print("Webcam not detected. Streaming synthetic demonstration feed.")
        frame_id = 0
        from create_dataset import create_face_image
        while True:
            frame_id += 1
            mask_flag = (frame_id // 40) % 2 == 0
            raw_face = create_face_image(with_mask=mask_flag, seed=frame_id % 50)
            raw_face = cv2.resize(raw_face, (400, 400))
            
            if net is not None:
                processed = preprocess_face(raw_face)
                preds = net.predict(processed)[0]
                pred_idx = np.argmax(preds)
                conf = float(preds[pred_idx]) * 100
                lbl = CLASS_LABELS[pred_idx]
                col = (0, 255, 0) if pred_idx == 0 else (0, 0, 255)
                
                cv2.rectangle(raw_face, (50, 50), (350, 350), col, 3)
                cv2.putText(raw_face, f"{lbl} ({conf:.1f}%)", (60, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, col, 2)
            
            ret, buffer = cv2.imencode('.jpg', raw_face)
            frame_bytes = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
            cv2.waitKey(100)
    else:
        while True:
            success, frame = cap.read()
            if not success:
                break
            else:
                faces = []
                if face_cascade is not None:
                    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(60, 60))
                
                if len(faces) == 0:
                    # Treat center portion as face if cascade not triggered
                    h, w, _ = frame.shape
                    faces = [(w//4, h//4, w//2, h//2)]
                    
                for (x, y, w, h) in faces:
                    face_roi = frame[y:y+h, x:x+w]
                    if face_roi.size > 0 and net is not None:
                        processed = preprocess_face(face_roi)
                        preds = net.predict(processed)[0]
                        pred_idx = np.argmax(preds)
                        conf = float(preds[pred_idx]) * 100
                        lbl = CLASS_LABELS[pred_idx]
                        color = (0, 255, 0) if pred_idx == 0 else (0, 0, 255)
                        
                        cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
                        cv2.putText(frame, f"{lbl}: {conf:.1f}%", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
                
                ret, buffer = cv2.imencode('.jpg', frame)
                frame_bytes = buffer.tobytes()
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    get_model()
    app.run(host='0.0.0.0', port=5000, debug=False)
