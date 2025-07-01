# 🔊 EchoID — Intelligent Speaker Recognition System

**EchoID** is a powerful and flexible speaker recognition system built with Resemblyzer. It identifies, compares, and learns speaker profiles over time using deep voice embeddings. With support for both `.mp3` and `.wav` audio, EchoID continually evolves by incorporating new samples, making it ideal for research, personal assistants, or voice-enabled security systems.

## 🚀 Key Features

### 🎤 Core Functionalities:

- **Speaker Identification**: Accurately identify voices from stored profiles.
- **New Speaker Registration**: Seamlessly add new voices when unidentified audio is detected.
- **Speaker Comparison**: Match new recordings against existing users with detailed confidence metrics.
- **Continual Learning**: System updates itself with new embeddings to improve accuracy over time.
- **Persistent JSON Storage**: Voice profiles are stored in structured JSON format.

### 🧠 Intelligence Layer:

- **Cosine Similarity Matching**: Uses a tunable threshold to improve speaker match precision.
- **Duplicate Prevention**: Avoids redundant data using distance-based filtering.
- **Confidence Scoring**: Match results come with dynamic confidence levels in percentage.

### 📊 In Progress / Planned:

- **Noise Robustness**: Integration with RNNoise or noise-reduction libraries.
- **Batch Processing**: Process and analyze multiple audio files in one go.
- **Streamlit GUI**: User-friendly web interface for easier interaction.
- **Charts & Visuals**: Graphical feedback for similarity and confidence results.
- **REST API Access**: Serve embeddings and recognition as a backend service.

## 🔧 Tech Stack

- Python
- Resemblyzer
- Pydub & FFMPEG
- NumPy & SciPy
- Streamlit (GUI - in progress)
- JSON for storage

## 📆 Ideal Use Cases

- Speaker verification in security systems
- Personalized voice assistants
- Educational or research projects
- Voice-controlled smart systems


> “EchoID learns the way you speak — and remembers it”
