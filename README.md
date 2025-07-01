🔊 EchoID — Intelligent Speaker Recognition System
EchoID is an advanced speaker recognition system powered by Resemblyzer, designed for both real-time and batch voice analysis. It leverages machine learning-based speaker embeddings and cosine similarity to identify, compare, and learn speaker profiles over time.

🧠 Key Features
  ✅ Core Functionality:
      🎤 Speaker Identification: Identify known speakers from voice input using cosine similarity on Resemblyzer embeddings.
      🗂️ Profile-based Memory: Store speaker profiles in structured JSON format with support for multiple audio samples per user
      🔁 Continual Learning: Automatically updates profiles when new voice data from a known speaker is detected.
      ➕ New Speaker Registration: Seamlessly add unknown speakers to the system.
      📏 Accurate Comparison Mode: Compare any two speaker recordings with confidence scoring.
      🧠 Duplicate Prevention: Avoid redundant embeddings using distance-based thresholding.
      
  🔜 Upcoming Additions:
      📊 Visual Feedback: Confidence charts for matching results and profile insights.
      🧼 Noise Robustness: Integrate noise reduction (e.g., RNNoise or noisereduce) before embedding.
      🖼️ Web GUI with Streamlit: Clean, interactive browser interface to upload, test, compare voices, and visualize results.
      📁 Batch Mode: Analyze multiple files at once for speaker matching or clustering.
      🧪 Confidence Logs & Testing: Add unit tests and a log for tracking match confidence across multiple sessions.

📦 Tech Stack
    Component	Technology
    Voice Embeddings	Resemblyzer
    Audio Preprocessing	Pydub, NumPy
    Similarity Metric	Cosine Similarity (scipy)
    Data Storage	JSON (local)
    GUI (planned)	Streamlit (Browser UI)
    Optional Enhancements	RNNoise, noisereduce (for denoising)

🧪 Example Use Cases
    Voice login for multi-user system
    Speaker comparison for forensic audio
    Interactive speaker recognition demo for teaching ML/audio

