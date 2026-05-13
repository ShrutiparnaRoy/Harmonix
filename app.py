from flask import Flask, render_template, request, jsonify, redirect, url_for, flash, session
import numpy as np
import librosa
import joblib
import soundfile as sf
import tempfile
import os
import pandas as pd
import time
import random
import smtplib
from email.message import EmailMessage
from flask_login import login_user, current_user, logout_user, login_required
from extensions import db, login_manager, bcrypt
from models import User, History

app = Flask(__name__)
app.config['SECRET_KEY'] = '5791628bb0b13ce0c676dfde280ba245'
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///site.db'

db.init_app(app)
login_manager.init_app(app)
bcrypt.init_app(app)
login_manager.login_view = 'login'

# Load ML components
try:
    print("Loading ML models...")
    start_time = time.time()
    model = joblib.load('best_model.pkl')
    scaler = joblib.load('scaler.pkl')
    label_encoder = joblib.load('label_encoder.pkl')
    print(f"ML models loaded in {time.time() - start_time:.2f}s")
except Exception as e:
    print(f"Warning: ML models not loaded. {e}")
    model = None
    scaler = None
    label_encoder = None

# Feature extraction (modified to take audio array and sr)
def extract_features(y, sr):
    try:
        start_feat = time.time()
        if y.ndim > 1:
            y = np.mean(y, axis=1)

        target_length = sr * 30
        if len(y) == 0:
            raise ValueError("Audio data is empty or corrupted")
        elif len(y) < target_length:
            y = np.pad(y, (0, target_length - len(y)), mode='constant')
        elif len(y) > target_length:
            y = y[:target_length]

        harmonic, _ = librosa.effects.hpss(y)

        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40)
        mfcc_delta = librosa.feature.delta(mfcc)
        mfcc_delta2 = librosa.feature.delta(mfcc, order=2)
        chroma = librosa.feature.chroma_stft(y=harmonic, sr=sr)
        mel = librosa.feature.melspectrogram(y=y, sr=sr)
        contrast = librosa.feature.spectral_contrast(y=harmonic, sr=sr)
        tonnetz = librosa.feature.tonnetz(y=harmonic, sr=sr)

        def stats(x):
            return np.hstack([np.mean(x, axis=1),
                              np.std(x, axis=1),
                              np.median(x, axis=1)]) if x.size != 0 else np.array([])

        features = np.hstack([
            stats(mfcc), stats(mfcc_delta), stats(mfcc_delta2),
            stats(chroma), stats(mel),
            stats(contrast), stats(tonnetz)
        ])

        expected_feature_length = scaler.n_features_in_ if hasattr(scaler, 'n_features_in_') else 258
        if features.shape[0] != expected_feature_length:
            print(f"Feature shape mismatch: {features.shape[0]}")

        print(f"Feature extraction took {time.time() - start_feat:.2f}s")
        return features
    except Exception as e:
        print(f"Error processing audio: {str(e)}")
        return None

# Krumhansl-Schmuckler Key-Finding Profiles
MAJOR_PROFILE = np.array([6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52, 5.19, 2.39, 3.66, 2.29, 2.88])
MINOR_PROFILE = np.array([6.33, 2.68, 3.52, 5.38, 2.60, 3.53, 2.54, 4.75, 3.98, 2.69, 3.34, 3.17])
PITCH_CLASSES = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']

def detect_key(y, sr):
    try:
        start_key = time.time()
        # Use HPSS and CQT for accuracy, but on the short snippet
        y_harmonic, _ = librosa.effects.hpss(y)
        chroma = librosa.feature.chroma_cqt(y=y_harmonic, sr=sr)
        
        # Sum chroma vectors over time
        chroma_sum = np.sum(chroma, axis=1)
        
        # Normalize
        if np.max(chroma_sum) > 0:
            chroma_sum = chroma_sum / np.max(chroma_sum)
        
        correlations = []
        
        # Calculate correlation for each major and minor key
        for i in range(12):
            # Rotate profiles to match the key tonic
            major_rotated = np.roll(MAJOR_PROFILE, i)
            minor_rotated = np.roll(MINOR_PROFILE, i)
            
            # Correlation
            if np.std(chroma_sum) > 0: # Avoid division by zero in correlation
                corr_major = np.corrcoef(chroma_sum, major_rotated)[0, 1]
                corr_minor = np.corrcoef(chroma_sum, minor_rotated)[0, 1]
            else:
                corr_major = 0
                corr_minor = 0
            
            correlations.append((corr_major, f"{PITCH_CLASSES[i]} Major"))
            correlations.append((corr_minor, f"{PITCH_CLASSES[i]} Minor"))
            
        # Find best match
        best_match = max(correlations, key=lambda x: x[0])
        print(f"Key detection took {time.time() - start_key:.2f}s")
        return best_match[1]
    except Exception as e:
        print(f"Error in key detection: {e}")
        return "Unknown"

def detect_chords(y, sr):
    try:
        start_chords = time.time()
        # Use HPSS and CQT for accuracy, but on the short snippet
        y_harmonic, _ = librosa.effects.hpss(y)
        chroma = librosa.feature.chroma_cqt(y=y_harmonic, sr=sr)
        
        # Define chord templates
        templates = {}
        for i, root in enumerate(PITCH_CLASSES):
            # Major
            vec = np.zeros(12)
            vec[i] = 1; vec[(i+4)%12] = 1; vec[(i+7)%12] = 1
            templates[f"{root} Major"] = vec / np.linalg.norm(vec)
            
            # Minor
            vec = np.zeros(12)
            vec[i] = 1; vec[(i+3)%12] = 1; vec[(i+7)%12] = 1
            templates[f"{root} Minor"] = vec / np.linalg.norm(vec)
            
        # Frame-wise chord detection
        chroma = librosa.util.normalize(chroma, axis=0)
        
        frames = chroma.shape[1]
        frame_time = librosa.frames_to_time(np.arange(frames), sr=sr)
        
        current_chord = None
        start_time_chord = 0
        results = []
        
        for t in range(frames):
            frame_chroma = chroma[:, t]
            
            best_score = -1
            best_chord = "N.C."
            
            if np.sum(frame_chroma) > 0.1: 
                for chord_name, template in templates.items():
                    score = np.dot(frame_chroma, template)
                    if score > best_score:
                        best_score = score
                        best_chord = chord_name
            
            if best_chord != current_chord:
                if current_chord is not None:
                    results.append({
                        "chord": current_chord,
                        "start": round(start_time_chord, 2),
                        "end": round(frame_time[t], 2)
                    })
                current_chord = best_chord
                start_time_chord = frame_time[t]
                
        if current_chord is not None:
             results.append({
                "chord": current_chord,
                "start": round(start_time_chord, 2),
                "end": round(frame_time[-1], 2)
            })
            
        # Filter very short chords
        final_results = [r for r in results if (r['end'] - r['start']) > 0.2]
        print(f"Chord detection took {time.time() - start_chords:.2f}s")
        return final_results
    except Exception as e:
        print(f"Error in chord detection: {e}")
        return []

@app.route('/')
@login_required
def index():
    return render_template('index.html')

@app.route('/login', methods=['GET', 'POST'])
def login():
    if current_user.is_authenticated:
        return redirect(url_for('index'))
    if request.method == 'POST':
        if 'email' in request.form:
            # Email entry (Step 1)
            email = request.form.get('email')
            otp = "".join([str(random.randint(0, 9)) for _ in range(6)])
            session['otp'] = otp
            session['temp_email'] = email

            msg = EmailMessage()
            msg.set_content(f'Your OTP for Harmonix is: {otp}')
            msg['Subject'] = 'Harmonix OTP Verification'
            msg['From'] = 'srgtesting2004@gmail.com'
            msg['To'] = email

            try:
                server = smtplib.SMTP('smtp.gmail.com', 587)
                server.starttls()
                server.login('srgtesting2004@gmail.com', 'dmko fouw kjdu qomh')
                server.send_message(msg)
                server.quit()
                flash('An OTP has been sent to your email.', 'info')
                return redirect(url_for('verify'))
            except Exception as e:
                flash('Failed to send OTP. Please try again.', 'danger')
        else:
            # Username/Password entry (Step 4 fallback)
            username = request.form.get('username')
            password = request.form.get('password')
            user = User.query.filter_by(username=username).first()
            if user and bcrypt.check_password_hash(user.password, password):
                login_user(user)
                return redirect(url_for('index'))
            else:
                flash('Login Unsuccessful. Please check username and password', 'danger')
    return render_template('login.html')

@app.route('/verify', methods=['GET', 'POST'])
def verify():
    if 'temp_email' not in session:
        return redirect(url_for('login'))
    if request.method == 'POST':
        user_otp = request.form.get('otp')
        if user_otp == session.get('otp'):
            email = session.get('temp_email')
            user = User.query.filter_by(email=email).first()
            if user:
                login_user(user)
                session.pop('otp', None)
                session.pop('temp_email', None)
                return redirect(url_for('index'))
            else:
                # New user, go to setup profile
                return redirect(url_for('setup_profile'))
        else:
            flash('Invalid OTP. Access Denied.', 'danger')
    return render_template('verify.html')

@app.route('/setup-profile', methods=['GET', 'POST'])
def setup_profile():
    if 'temp_email' not in session:
        return redirect(url_for('login'))
    if request.method == 'POST':
        username = request.form.get('username')
        password = request.form.get('password')
        genres = request.form.get('favourite_genres')
        artists = request.form.get('favourite_artists')
        hashed_password = bcrypt.generate_password_hash(password).decode('utf-8')
        user = User(email=session['temp_email'], username=username, password=hashed_password, 
                    favourite_genres=genres, favourite_artists=artists)
        db.session.add(user)
        db.session.commit()
        login_user(user)
        session.pop('otp', None)
        session.pop('temp_email', None)
        flash('Profile created successfully!', 'success')
        return redirect(url_for('index'))
    return render_template('setup_profile.html')

@app.route('/logout')
def logout():
    logout_user()
    flash('You have been logged out. You must verify your email again to sign back in via OTP.', 'info')
    return redirect(url_for('login'))

@app.route('/history')
@login_required
def history():
    user_history = History.query.filter_by(user_id=current_user.id).order_by(History.timestamp.desc()).all()
    return render_template('history.html', history=user_history, user=current_user)

@app.route('/metronome')
@login_required
def metronome():
    return render_template('metronome.html')

@app.route('/pitch-tuner')
@login_required
def pitch_detector():
    return render_template('pitch-tuner.html')

@app.route('/rhythm-detector')
@login_required
def rhythm_detector():
    return render_template('rhythm-detector.html')

@app.route('/analyze-rhythm', methods=['POST'])
@login_required
def analyze_rhythm():
    if 'audio_file' not in request.files: return jsonify({'error': 'No audio file provided'})
    audio_file = request.files['audio_file']
    if audio_file.filename == '': return jsonify({'error': 'No file selected'})
    try:
        start_time_rhythm = time.time()
        with tempfile.NamedTemporaryFile(delete=False, suffix=audio_file.filename) as temp_file:
            audio_file.save(temp_file.name)
            temp_file_path = temp_file.name
        
        # Load with sr=None to ensure compatibility
        y, sr = librosa.load(temp_file_path, sr=None, duration=20)
        print(f"Audio loaded for rhythm in {time.time() - start_time_rhythm:.2f}s, sr={sr}")
        
        onset_env = librosa.onset.onset_strength(y=y, sr=sr)
        tempo = librosa.beat.tempo(onset_envelope=onset_env, sr=sr)
        tempo_val = float(tempo[0])
        res_str = f"Tempo: {tempo_val:.1f} BPM"
        hist = History(activity_type='Rhythm Detection', filename=audio_file.filename, result=res_str, author=current_user)
        db.session.add(hist)
        db.session.commit()
        print(f"Total rhythm analysis took {time.time() - start_time_rhythm:.2f}s")
        return jsonify({'tempo': round(tempo_val, 1), 'time_signature': "4/4", 'message': 'Rhythm analysis successful'})
    except Exception as e: 
        print(f"Rhythm Error: {e}")
        return jsonify({'error': str(e)})
    finally:
         if 'temp_file_path' in locals() and os.path.exists(temp_file_path): os.remove(temp_file_path)

@app.route('/predict', methods=['POST'])
@login_required
def predict():
    if 'audio_file' not in request.files: return jsonify({'error': 'No audio file provided'})
    audio_file = request.files['audio_file']
    if audio_file.filename == '': return jsonify({'error': 'No file selected'})
    start_time_req = float(request.form.get('start_time', 0))
    end_time_req = float(request.form.get('end_time', 30))
    try:
        start_time_predict = time.time()
        with tempfile.NamedTemporaryFile(delete=False, suffix=audio_file.filename) as temp_file:
            audio_file.save(temp_file.name)
            temp_file_path = temp_file.name
        
        # Load for prediction
        duration = end_time_req - start_time_req
        y, sr = librosa.load(temp_file_path, sr=None, mono=True, offset=start_time_req, duration=duration)
        print(f"Audio loaded for prediction in {time.time() - start_time_predict:.2f}s, sr={sr}")
        
        features = extract_features(y, sr)
        if features is None: return jsonify({'error': 'Feature extraction failed'})
        if model is None: return jsonify({'error': 'Model not loaded'})
        
        features_scaled = scaler.transform([features])
        prediction = model.predict(features_scaled)
        predicted_genre = label_encoder.inverse_transform(prediction)[0]
        
        hist = History(activity_type='Genre Prediction', filename=audio_file.filename, result=predicted_genre, author=current_user)
        db.session.add(hist)
        db.session.commit()
        
        probabilities = {}
        if hasattr(model, 'predict_proba'):
            probs = model.predict_proba(features_scaled)[0]
            for i, genre in enumerate(label_encoder.classes_): probabilities[genre] = float(probs[i])
        else:
            for i, genre in enumerate(label_encoder.classes_): probabilities[genre] = 1.0 if i == prediction[0] else 0.0
        
        print(f"Total genre prediction took {time.time() - start_time_predict:.2f}s")
        return jsonify({'predicted_genre': predicted_genre, 'probabilities': probabilities})
    except Exception as e: 
        print(f"Predict Error: {e}")
        return jsonify({'error': f'Error processing audio: {str(e)}'})
    finally:
        if 'temp_file_path' in locals() and os.path.exists(temp_file_path): os.remove(temp_file_path)

@app.route('/scale-finder')
@login_required
def scale_finder():
    return render_template('scale-finder.html')

@app.route('/analyze-scale', methods=['POST'])
@login_required
def analyze_scale():
    if 'audio_file' not in request.files: return jsonify({'error': 'No audio file provided'})
    audio_file = request.files['audio_file']
    if audio_file.filename == '': return jsonify({'error': 'No file selected'})
    try:
        start_time_scale = time.time()
        with tempfile.NamedTemporaryFile(delete=False, suffix=audio_file.filename) as temp_file:
            audio_file.save(temp_file.name)
            temp_file_path = temp_file.name
        
        # Load for scale
        y, sr = librosa.load(temp_file_path, sr=None, duration=15)
        print(f"Audio loaded for scale in {time.time() - start_time_scale:.2f}s, sr={sr}")
        
        detected_key = detect_key(y, sr)
        hist = History(activity_type='Scale Finder', filename=audio_file.filename, result=f"Key: {detected_key}", author=current_user)
        db.session.add(hist)
        db.session.commit()
        print(f"Total scale analysis took {time.time() - start_time_scale:.2f}s")
        return jsonify({'key': detected_key, 'message': 'Scale analysis successful'})
    except Exception as e: 
        print(f"Scale Error: {e}")
        return jsonify({'error': str(e)})
    finally:
         if 'temp_file_path' in locals() and os.path.exists(temp_file_path): os.remove(temp_file_path)

@app.route('/blog')
@login_required
def blog():
    return render_template('blog.html')

@app.route('/chord-tracker')
@login_required
def chord_tracker():
    return render_template('chord-tracker.html')

@app.route('/analyze-chords', methods=['POST'])
@login_required
def analyze_chords():
    if 'audio_file' not in request.files: return jsonify({'error': 'No audio file provided'})
    audio_file = request.files['audio_file']
    if audio_file.filename == '': return jsonify({'error': 'No file selected'})
    try:
        start_time_chords = time.time()
        with tempfile.NamedTemporaryFile(delete=False, suffix=audio_file.filename) as temp_file:
            audio_file.save(temp_file.name)
            temp_file_path = temp_file.name
        
        # Load for chords
        y, sr = librosa.load(temp_file_path, sr=None, duration=20)
        print(f"Audio loaded for chords in {time.time() - start_time_chords:.2f}s, sr={sr}")
        
        chords = detect_chords(y, sr)
        chord_summary = ", ".join([c['chord'] for c in chords[:5]]) + "..." if chords else "No chords detected"
        hist = History(activity_type='Chord Tracker', filename=audio_file.filename, result=f"Chords: {chord_summary}", author=current_user)
        db.session.add(hist)
        db.session.commit()
        print(f"Total chord analysis took {time.time() - start_time_chords:.2f}s")
        return jsonify({'chords': chords, 'message': 'Chord analysis successful'})
    except Exception as e: 
        print(f"Chord Error: {e}")
        return jsonify({'error': str(e)})
    finally:
         if 'temp_file_path' in locals() and os.path.exists(temp_file_path): os.remove(temp_file_path)

if __name__ == '__main__':
    with app.app_context():
        db.create_all()
    app.run(debug=True)
