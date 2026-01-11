import os
import numpy as np
import librosa
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras import layers, models # type: ignore
from tensorflow.keras.utils import to_categorical # type: ignore
import whisper
from langdetect import detect
import tkinter as tk
from tkinter import filedialog


# -----------------------------
# STEP 1: Dataset Path
# -----------------------------
DATASET_PATH = "dataset/genre"  # Change this to your dataset folder

# -----------------------------
# STEP 2: Feature Extraction (for Genre Classification)
# -----------------------------
def extract_features(file_path, max_pad_len=128):
    try:
        audio, sr = librosa.load(file_path, duration=30)  # Load 30 sec
        mel_spec = librosa.feature.melspectrogram(y=audio, sr=sr, n_mels=128)
        log_mel_spec = librosa.power_to_db(mel_spec, ref=np.max)

        # Pad or truncate to fixed length
        if log_mel_spec.shape[1] < max_pad_len:
            pad_width = max_pad_len - log_mel_spec.shape[1]
            log_mel_spec = np.pad(log_mel_spec, pad_width=((0,0),(0,pad_width)), mode='constant')
        else:
            log_mel_spec = log_mel_spec[:, :max_pad_len]

        return log_mel_spec
    except Exception as e:
        print("Error extracting features:", e)
        return None

# -----------------------------
# STEP 3: Prepare Dataset
# -----------------------------
genres = []
features = []

for genre in os.listdir(DATASET_PATH):
    genre_path = os.path.join(DATASET_PATH, genre)
    if os.path.isdir(genre_path):
        for file in os.listdir(genre_path):
            file_path = os.path.join(genre_path, file)

            # ✅ Only process audio files
            if not file_path.lower().endswith(('.wav', '.mp3', '.flac')):
                continue

            data = extract_features(file_path)
            if data is not None:
                features.append(data)
                genres.append(genre)


X = np.array(features)
X = X.reshape(X.shape[0], 128, 128, 1)  # CNN input shape

# Encode genre labels
encoder = LabelEncoder()
y = encoder.fit_transform(genres)
y = to_categorical(y, num_classes=len(set(genres)))

# Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# -----------------------------
# STEP 4: Build CNN Model
# -----------------------------
model = models.Sequential([
    layers.Conv2D(32, (3,3), activation='relu', input_shape=(128,128,1)),
    layers.MaxPooling2D((2,2)),
    layers.Conv2D(64, (3,3), activation='relu'),
    layers.MaxPooling2D((2,2)),
    layers.Conv2D(128, (3,3), activation='relu'),
    layers.MaxPooling2D((2,2)),
    layers.Flatten(),
    layers.Dense(128, activation='relu'),
    layers.Dropout(0.3),
    layers.Dense(len(set(genres)), activation='softmax')
])

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
model.summary()

# -----------------------------
# STEP 5: Train Model
# -----------------------------
history = model.fit(
    X_train, y_train,
    epochs=25, batch_size=32,
    validation_split=0.2, verbose=1
)

# -----------------------------
# STEP 6: Evaluate Model
# -----------------------------
test_loss, test_acc = model.evaluate(X_test, y_test, verbose=2)
print(f"\n🎯 Test Accuracy: {test_acc * 100:.2f}%")

# -----------------------------
# STEP 7: Plot Accuracy & Loss
# -----------------------------
plt.figure(figsize=(12,5))

# Accuracy
plt.subplot(1,2,1)
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title("Model Accuracy")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.legend()

# Loss
plt.subplot(1,2,2)
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title("Model Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()

plt.show()

# -----------------------------
# File Selection
# -----------------------------
root = tk.Tk()
root.withdraw()  # Hide the main Tk window

file_path = filedialog.askopenfilename(
    title="🎵 Select a Music File",
    filetypes=[("Audio Files", "*.mp3 *.wav *.flac")]
)

if not file_path:
    print("❌ No file selected. Exiting...")
    exit()
else:
    print(f"✅ File selected: {file_path}")

# -----------------------------
# STEP 8: Predict Genre + Language
# -----------------------------
# Load Whisper model once
whisper_model = whisper.load_model("base")

# Language mapping (ISO -> Full name)
LANGUAGE_MAP = {
    "en": "English",
    "ta": "Tamil",
    "hi": "Hindi",
    "te": "Telugu",
    "ml": "Malayalam",
    "fr": "French",
    "es": "Spanish",
    "unknown": "Unknown"
}

def predict_genre_and_language(file_path):
    # Normalize + absolute path for Windows
    file_path = os.path.normpath(file_path)
    abs_path = os.path.abspath(file_path)

    # Debugging: check if file exists
    if not os.path.exists(abs_path):
        print(f"❌ File not found: {abs_path}")
        return "unknown", "unknown", "..."

    # Predict genre
    feature = extract_features(abs_path)
    predicted_genre = "unknown"
    if feature is not None:
        feature = feature.reshape(1, 128, 128, 1)
        prediction = model.predict(feature)
        predicted_genre = encoder.inverse_transform([np.argmax(prediction)])[0]

    # Detect language using Whisper
    try:
        result = whisper_model.transcribe(abs_path)  # ✅ absolute path used here
        lyrics_text = result.get("text", "")
        lang_code = result.get("language", "unknown")

        # Map language code to full name if possible
        detected_lang = LANGUAGE_MAP.get(lang_code, lang_code)

    except Exception as e:
        print("Whisper error:", e)
        detected_lang, lyrics_text = "Unknown", "..."

    return predicted_genre, detected_lang, lyrics_text



# Example usage
custom_file = file_path
genre, language, lyrics = predict_genre_and_language(custom_file)
print(f"\nPredicted Genre: {genre}")
print(f"Detected Language: {language}")
print(f"Transcribed Lyrics (preview): {lyrics[:100]}...")