import cv2
import numpy as np
import tensorflow as tf
from deepface import DeepFace
from collections import Counter, deque
import time

# Enable TensorFlow GPU for DeepFace
for gpu in tf.config.experimental.list_physical_devices('GPU'):
    tf.config.experimental.set_memory_growth(gpu, True)

# Load YuNet face detection model (CPU only)
face_detector = cv2.FaceDetectorYN.create("face_detection_yunet_2023mar.onnx", "", (640, 480))

# Initialize webcam
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
if not cap.isOpened():
    print("❌ Error: Could not open webcam.")
    exit()

# Define analyze_emotions function before it's referenced
def analyze_emotions(emotion_history):
    """
    Advanced emotion analysis that accounts for:
    1. Non-neutral emotions having more significance
    2. High confidence emotions carrying more weight
    3. Duration of emotions
    """
    if not emotion_history:
        print("⚠️ No emotions recorded.")
        return
    
    # Create basic counters
    emotion_counts = Counter()
    for entry in emotion_history:
        emotion_counts[entry['emotion']] += 1
    
    # Calculate time spent in each emotion (approximate)
    total_entries = len(emotion_history)
    time_per_entry = 10 / total_entries if total_entries > 0 else 0
    emotion_durations = {emotion: count * time_per_entry for emotion, count in emotion_counts.items()}
    
    # Create weighted analysis that prioritizes non-neutral emotions
    weighted_emotions = {}
    neutral_weight = 0.3  # Further reduced neutral weight to prioritize other emotions
    
    for entry in emotion_history:
        emotion = entry['emotion']
        confidence = entry['confidence']
        weight = neutral_weight if emotion == 'neutral' else 1.5  # Increased weight for non-neutral emotions
        if emotion not in weighted_emotions:
            weighted_emotions[emotion] = 0
        weighted_emotions[emotion] += weight * confidence
    
    # Find most common, most significant, and highest weighted emotions
    most_frequent = emotion_counts.most_common(1)[0][0] if emotion_counts else None
    longest_duration = max(emotion_durations.items(), key=lambda x: x[1])[0] if emotion_durations else None
    most_significant = max(weighted_emotions.items(), key=lambda x: x[1])[0] if weighted_emotions else None
    
    # Print detailed analysis
    print("\n📊 Emotion Analysis Results:")
    print(f"Total frames analyzed: {len(emotion_history)}")
    
    print("\n⏱️ Approximate duration of each emotion:")
    for emotion, duration in sorted(emotion_durations.items(), key=lambda x: x[1], reverse=True):
        percentage = (duration / 10) * 100
        print(f"  {emotion}: {duration:.1f}s ({percentage:.1f}%)")
    
    print("\n🔍 Analysis based on different methods:")
    print(f"  Most frequent emotion: {most_frequent}")
    print(f"  Longest duration emotion: {longest_duration}")
    print(f"  Most significant emotion (weighted by intensity and non-neutrality): {most_significant}")
    
    # Make final recommendation with even stronger bias against neutral
    if most_significant != 'neutral' and weighted_emotions.get(most_significant, 0) > weighted_emotions.get('neutral', 0) * 0.6:  # Reduced threshold
        print(f"\n✅ Recommended emotion to consider: {most_significant}")
        print("   (This accounts for brief but meaningful emotional responses)")
    else:
        print(f"\n✅ Recommended emotion to consider: {longest_duration}")
        print("   (This represents the dominant emotional state during recording)")

# Emotion tracking with improved logic
recording = False
emotion_history = deque()  # Store emotion data with timestamps
start_time = 0
recording_duration = 10  # Duration in seconds

print("\n📌 Press 's' to START recording emotions for 10 seconds.\n📌 Press 'e' to manually STOP recording and see results.\n📌 Press 'q' to EXIT.\n")

while True:
    ret, frame = cap.read()
    if not ret:
        print("❌ Error: Failed to capture image.")
        break

    current_time = time.time()
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    _, faces = face_detector.detect(frame)
    
    # Stop recording automatically after 10 seconds
    if recording and (current_time - start_time >= recording_duration):
        recording = False
        print("⏹️ Recording stopped automatically after 10 seconds.")
        analyze_emotions(emotion_history)

    if faces is not None:
        for face in faces:
            x, y, w, h = map(int, face[:4])
            face_roi = rgb_frame[y:y+h, x:x+w]
            try:
                result = DeepFace.analyze(face_roi, actions=['emotion'], enforce_detection=False)
                if result:
                    emotions = result[0]['emotion']
                    
                    # Boosted emotion recognition for non-neutral emotions
                    boost_factor = {
                        "sad": 1.6,      # Increased from 1.25
                        "happy": 1.6,    # New boost for happy
                        "surprise": 1.5, # New boost for surprise
                        "angry": 1.5,    # New boost for angry
                        "disgust": 1.5,  # Increased from 1.25
                        "fear": 1.5      # New boost for fear
                    }
                    
                    # Apply boost factors to all emotions except neutral
                    updated_emotions = {}
                    for emotion, score in emotions.items():
                        if emotion == "neutral":
                            updated_emotions[emotion] = score * 0.75  # Reduce neutral scores by 25%
                        else:
                            updated_emotions[emotion] = score * boost_factor.get(emotion, 1.0)
                    
                    dominant_emotion = max(updated_emotions, key=updated_emotions.get)
                    if recording:
                        emotion_history.append({
                            'timestamp': current_time - start_time,
                            'emotion': dominant_emotion,
                            'confidence': updated_emotions[dominant_emotion],
                            'all_emotions': updated_emotions
                        })
                    
                    # Display current emotion with confidence score
                    emotion_text = f"{dominant_emotion}: {updated_emotions[dominant_emotion]:.1f}"
                    cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                    cv2.putText(frame, emotion_text, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                    
                    # Show top 3 emotions in the corner of the screen (helpful for debugging)
                    sorted_emotions = sorted(updated_emotions.items(), key=lambda x: x[1], reverse=True)[:3]
                    for i, (emotion, score) in enumerate(sorted_emotions):
                        cv2.putText(frame, f"{emotion}: {score:.1f}", (10, 60 + i*25), 
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)
                        
            except Exception as e:
                print(f"⚠️ Error analyzing emotion: {e}")

    # Display recording status and timer
    if recording:
        elapsed = current_time - start_time
        remaining = max(0, recording_duration - elapsed)
        status_text = f"Recording: {remaining:.1f}s left"
        cv2.putText(frame, status_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

    cv2.imshow("Live Emotion Detection", frame)
    key = cv2.waitKey(1) & 0xFF
    if key == ord('s'):
        recording = True
        emotion_history.clear()
        start_time = time.time()
        print("▶️ Started recording emotions for 10 seconds.")
    elif key == ord('e'):
        if recording:
            recording = False
            print("⏹️ Recording stopped manually.")
            analyze_emotions(emotion_history)
    elif key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()