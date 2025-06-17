import os
from flask import Flask, render_template, Response, request, redirect, url_for, flash, session
import cv2
import numpy as np
import mediapipe as mp
from flask_sqlalchemy import SQLAlchemy
from werkzeug.security import generate_password_hash, check_password_hash
import os
import shutil
from flask import jsonify

app = Flask(__name__)
app.config['SECRET_KEY'] = 'your-secret-key'
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///users.db'
app.config['MAX_CONTENT_LENGTH'] = 100 * 1024 * 1024  # 100MB limit
db = SQLAlchemy(app)

# User model for database storage
class User(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    fullname = db.Column(db.String(100), nullable=False)
    username = db.Column(db.String(50), unique=True, nullable=False)
    email = db.Column(db.String(100), nullable=False)
    phone = db.Column(db.String(20))
    password = db.Column(db.String(200), nullable=False)

# Global variables for exercise counts
pushup_count = 0
squat_count = 0
jump_count = 0
curl_count = 0

# State tracking for repetitions
pushup_state = None
squat_state = None
jump_state = None
curl_state = None

# Webcam status flag
webcam_active = True  # Controls whether webcam runs

# MediaPipe setup for pose estimation
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)
mp_draw = mp.solutions.drawing_utils

# Video capture
cap = cv2.VideoCapture(0)

def restart_webcam():
    """Restart webcam if it was stopped"""
    global webcam_active, cap
    if not webcam_active:
        cap = cv2.VideoCapture(0)  # Restart video capture
        webcam_active = True
        print("Webcam restarted")

def calculate_angle(a, b, c):
    """Calculate angle between three points"""
    a, b, c = np.array(a), np.array(b), np.array(c)
    radians = np.arctan2(c[1]-b[1], c[0]-b[0]) - np.arctan2(a[1]-b[1], a[0]-b[0])
    angle = np.abs(radians * 180.0 / np.pi)
    return angle if angle <= 180.0 else 360 - angle


def detect_exercise(landmarks, exercise_type):
    """Detect exercise movements and count repetitions"""
    global pushup_count, squat_count, jump_count, curl_count
    global pushup_state, squat_state, jump_state, curl_state

    if exercise_type == 'pushup':
        left_shoulder = [landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x,
                         landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y]
        left_elbow = [landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].x,
                      landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].y]

        current_state = "down" if left_elbow[1] > left_shoulder[1] else "up"
        if pushup_state == "down" and current_state == "up":
            pushup_count += 1
        pushup_state = current_state

    elif exercise_type == 'squat':
        left_hip = landmarks[mp_pose.PoseLandmark.LEFT_HIP.value]
        left_knee = landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value]
        left_ankle = landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value]
        left_shoulder = landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value]

        hip_angle = calculate_angle(
            [landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x, landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y],
            [left_hip.x, left_hip.y],
            [left_knee.x, left_knee.y]
        )

        knee_angle = calculate_angle(
            [left_hip.x, left_hip.y],
            [left_knee.x, left_knee.y],
            [left_ankle.x, left_ankle.y]
        )

        if hip_angle < 160 and knee_angle < 100:
            current_state = "down"
        else:
            current_state = "up"

        if squat_state == "down" and current_state == "up":
            squat_count += 1
        squat_state = current_state

    elif exercise_type == 'jump':
        # Get landmarks for key body points
        left_heel = landmarks[mp_pose.PoseLandmark.LEFT_HEEL.value].y
        right_heel = landmarks[mp_pose.PoseLandmark.RIGHT_HEEL.value].y
        left_ankle = landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].y
        right_ankle = landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value].y

        # Calculate average heel position
        avg_heel_y = (left_heel + right_heel) / 2

        # Determine the state based on heel position
        if avg_heel_y < 0.7:  # Adjust threshold as needed
            current_state = "air"
        else:
            current_state = "ground"

        # Count repetitions
        if jump_state == "air" and current_state == "ground":
            jump_count += 1

        jump_state = current_state

    elif exercise_type == 'curl':
        # Get landmarks for key body points
        left_shoulder = [landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x,
                         landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y]
        left_elbow = [landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].x,
                      landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].y]
        left_wrist = [landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].x,
                      landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].y]

        # Calculate the angle at the elbow
        angle = calculate_angle(left_shoulder, left_elbow, left_wrist)

        # Determine the state based on the angle
        if angle < 40:
            current_state = "up"
        else:
            current_state = "down"

        # Count repetitions
        if curl_state == "down" and current_state == "up":
            curl_count += 1

        curl_state = current_state


def gen_frames(exercise_type):
    """Capture video frames and overlay exercise count"""
    global webcam_active
    while webcam_active:
        success, frame = cap.read()
        if not success:
            break

        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = pose.process(frame)

        if results.pose_landmarks:
            mp_draw.draw_landmarks(frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)
            detect_exercise(results.pose_landmarks.landmark, exercise_type)

        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

        # Overlay the counter onto the video frame
        count_position = (50, 400)
        count_value = {
            'pushup': pushup_count,
            'squat': squat_count,
            'jump': jump_count,
            'curl': curl_count
        }.get(exercise_type, 0)

        cv2.putText(frame, f'Count: {count_value}', count_position, 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 3, cv2.LINE_AA)

        ret, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

    cap.release()

# NEW: Add video_feed route
@app.route('/process_video/<exercise_type>')
def process_video_route(exercise_type):
    video_path = request.args.get('video_path')
    return Response(process_video(video_path, exercise_type),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/video_feed/<exercise_type>')
def video_feed(exercise_type):
    return Response(gen_frames(exercise_type),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

# Update the main route to point to the login page
@app.route('/')
def login():
    return render_template('login.html')

@app.route('/index')
def index():
    if 'username' not in session:
        return redirect(url_for('login'))
    return render_template('index.html', 
                            pushup_count=pushup_count, 
                            squat_count=squat_count, 
                            jump_count=jump_count, 
                            curl_count=curl_count)
@app.route('/pushup')
def pushup():
    if 'username' not in session:
        return redirect(url_for('login'))
    session['exercise_type'] = 'pushup'  # Store the current exercise type in the session
    restart_webcam()
    return render_template('pushup.html', count=pushup_count)

@app.route('/squat')
def squat():
    if 'username' not in session:
        return redirect(url_for('login'))
    session['exercise_type'] = 'squat'
    restart_webcam()
    return render_template('squat.html', count=squat_count)

@app.route('/jump')
def jump():
    if 'username' not in session:
        return redirect(url_for('login'))
    session['exercise_type'] = 'jump'
    restart_webcam()
    return render_template('jump.html', count=jump_count)

@app.route('/curl')
def curl():
    if 'username' not in session:
        return redirect(url_for('login'))
    session['exercise_type'] = 'curl'
    restart_webcam()
    return render_template('curl.html', count=curl_count)

@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        # Get form data
        fullname = request.form['fullname']
        username = request.form['username']
        email = request.form['email']
        phone = request.form['phone']
        password = request.form['password']
        confirm_password = request.form['confirm_password']

        # Basic validation
        if not username or not password:
            flash('Username and password are required')
            return redirect(url_for('register'))

        if password != confirm_password:
            flash('Passwords do not match')
            return redirect(url_for('register'))

        # Hash password using pbkdf2:sha256 method
        hashed_password = generate_password_hash(password, method='pbkdf2:sha256')

        # Check if username already exists
        existing_user = User.query.filter_by(username=username).first()
        if existing_user:
            flash('Username already exists')
            return redirect(url_for('register'))

        # Create new user
        new_user = User(
            fullname=fullname,
            username=username,
            email=email,
            phone=phone,
            password=hashed_password
        )

        db.session.add(new_user)
        db.session.commit()

        flash('Register successfully!')
        return redirect(url_for('login'))

    return render_template('register.html')

@app.route('/login', methods=['GET', 'POST'])
def login_route():
    if request.method == 'POST':
        # Get form data
        username = request.form['username']
        password = request.form['password']

        # Find user by username
        user = User.query.filter_by(username=username).first()

        # Verify user and password
        if user and check_password_hash(user.password, password):
            session['username'] = user.username
            return redirect(url_for('index'))
        else:
            flash('Invalid username or password')

    return render_template('login.html')

@app.route('/forgot_password', methods=['GET', 'POST'])
def forgot_password():
    if request.method == 'POST':
        # Get form data
        username = request.form['username']
        email = request.form['email']

        # Find user by username and email
        user = User.query.filter_by(username=username, email=email).first()

        if user:
            # In a real application, you would send a password reset email here
            flash('Password reset instructions have been sent to your email')
            return redirect(url_for('login'))
        else:
            flash('User not found')

    return render_template('forgot_password.html')

@app.route('/logout')
def logout():
    session.pop('username', None)
    return redirect(url_for('login'))

@app.route('/upload_video/<exercise_type>', methods=['POST'])
def upload_video(exercise_type):
    if 'username' not in session:
        return redirect(url_for('login'))
    
    if 'video' not in request.files:
        flash('No video file provided')
        return redirect(request.referrer)
    
    video_file = request.files['video']
    if video_file.filename == '':
        flash('No selected file')
        return redirect(request.referrer)
    
    # Check if the file size exceeds the allowed limit
    if len(video_file.read()) > app.config['MAX_CONTENT_LENGTH']:
        flash('File size exceeds the allowed limit')
        return redirect(request.referrer)
    
    # Reset the file pointer to the beginning after reading
    video_file.seek(0)
    
    # Create the directory if it doesn't exist
    video_dir = 'static/videos'
    os.makedirs(video_dir, exist_ok=True)
    
    # Save the video file
    video_path = os.path.join(video_dir, video_file.filename)
    video_file.save(video_path)
    
    # Redirect to the processing route
    return redirect(url_for('process_video_route', exercise_type=exercise_type, video_path=video_path))


@app.route('/upload_squat', methods=['POST'])
def upload_squat():
    if 'username' not in session:
        return redirect(url_for('login'))
    
    if 'video' not in request.files:
        return jsonify({'success': False, 'message': 'No video file provided'}), 400
    
    video_chunk = request.files['video']
    filename = request.form.get('filename')
    chunk = int(request.form.get('chunk'))
    total_chunks = int(request.form.get('totalChunks'))
    
    # Create temporary directory for chunks
    chunk_dir = 'static/chunks'
    os.makedirs(chunk_dir, exist_ok=True)
    
    # Save the chunk
    chunk_path = os.path.join(chunk_dir, f"{filename}_part_{chunk}")
    video_chunk.save(chunk_path)
    
    # If all chunks are uploaded, combine them and return the video path
    if chunk == total_chunks:
        video_path = combine_chunks(filename, total_chunks, chunk_dir)
        return jsonify({'success': True, 'video_path': video_path})
    
    return jsonify({'success': True})

def combine_chunks(filename, total_chunks, chunk_dir):
    video_dir = 'static/videos'
    os.makedirs(video_dir, exist_ok=True)
    video_path = os.path.join(video_dir, filename)
    
    with open(video_path, 'wb') as f:
        for i in range(1, total_chunks + 1):
            chunk_path = os.path.join(chunk_dir, f"{filename}_part_{i}")
            with open(chunk_path, 'rb') as chunk_f:
                f.write(chunk_f.read())
            os.remove(chunk_path)  # Clean up chunk files
    
    # Return the video path to the client
    return video_path
    

@app.route('/stop_webcam', methods=['POST'])
def stop_webcam():
    """Stop the webcam when the button is clicked"""
    global webcam_active
    webcam_active = False
    cap.release()
    return 'Webcam stopped', 200

# Add reset routes for each exercise
@app.route('/reset_pushup', methods=['POST'])
def reset_pushup():
    global pushup_count, webcam_active
    pushup_count = 0
    webcam_active = False
    cap.release()
    return 'Activity reset successfully'

@app.route('/reset_squat', methods=['POST'])
def reset_squat():
    global squat_count, webcam_active
    squat_count = 0
    webcam_active = False
    cap.release()
    return 'Activity reset successfully'

@app.route('/reset_jump', methods=['POST'])
def reset_jump():
    global jump_count, webcam_active
    jump_count = 0
    webcam_active = False
    cap.release()
    return 'Activity reset successfully'

@app.route('/reset_curl', methods=['POST'])
def reset_curl():
    global curl_count, webcam_active
    curl_count = 0
    webcam_active = False
    cap.release()
    return 'Activity reset successfully'

def process_video(video_path, exercise_type):
    """Process a video file and stream results"""
    global pushup_count, squat_count, jump_count, curl_count
    pushup_count = 0
    squat_count = 0
    jump_count = 0
    curl_count = 0
    
    # Initialize pose estimation inside the function
    with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
        cap = cv2.VideoCapture(video_path)
        
        while cap.isOpened():
            success, frame = cap.read()
            if not success:
                break
            
            # Convert the BGR image to RGB and process it
            results = pose.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            
            if results.pose_landmarks:
                mp_draw.draw_landmarks(frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)
                detect_exercise(results.pose_landmarks.landmark, exercise_type)
            
            # Overlay the counter onto the video frame
            count_position = (50, 400)
            count_value = {
                'pushup': pushup_count,
                'squat': squat_count,
                'jump': jump_count,
                'curl': curl_count
            }.get(exercise_type, 0)
            
            cv2.putText(frame, f'Count: {count_value}', count_position, 
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 3, cv2.LINE_AA)
            
            # Encode the frame as JPEG
            ret, buffer = cv2.imencode('.jpg', frame)
            if not ret:
                break
                
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')
        
        cap.release()

@app.route('/get_count/<exercise_type>')
def get_count(exercise_type):
    count_value = {
        'pushup': pushup_count,
        'squat': squat_count,
        'jump': jump_count,
        'curl': curl_count
    }.get(exercise_type, 0)
    return {'count': count_value}

# Add result templates for each exercise
@app.route('/result_pushup')
def result_pushup():
    return render_template('result_pushup.html')

@app.route('/result_squat')
def result_squat():
    return render_template('result_squat.html')

@app.route('/result_jump')
def result_jump():
    return render_template('result_jump.html')

@app.route('/result_curl')
def result_curl():
    return render_template('result_curl.html')

if __name__ == '__main__':
    with app.app_context():
        db.create_all()  # Create database tables
    app.run(debug=True)