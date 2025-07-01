'''DEVELOPER NOTICE:
                What we have done:
                        Speaker identification using Resemblyzer
                        Profile-based memory using JSON
                        Continual learning — updates speaker profiles as new audios are added
                        New speaker registration
                        Speaker comparison mode with accurate confidence score
                        Cosine similarity-based matching with an adjustable threshold
                        Duplicate prevention using distance-based filtering
                ADDITIONAL CONTENT:
                        Chart output in similarity and recognition
                        GUI Interface using Tkinter or PyQt or react and fast api
                        Noise Robustness : Apply noise filtering before processing
                        Visual Feedback : Show speaker match results with charts    
                        Batch Processing : Analyze multiple files at once'''



import json
from pathlib import Path
from resemblyzer import VoiceEncoder, preprocess_wav
from pydub import AudioSegment
import numpy as np
from scipy.spatial.distance import cosine

# ===================== Constants =====================
THRESHOLD = 0.20
PROFILE_PATH = Path("profiles.json")


# ===================== Audio Loading =====================
def load_file(file_path):
    """
    Loads .wav or .mp3 and returns a preprocessed waveform ready for embedding.
    """
    ext = file_path.suffix.lower()
    if ext == ".wav":
        return preprocess_wav(file_path)
    elif ext == ".mp3":
        audio = AudioSegment.from_mp3(file_path)
        audio = audio.set_channels(1).set_frame_rate(16000)
        samples = np.array(audio.get_array_of_samples()).astype(np.float32) / 32768.0
        return preprocess_wav(samples, source_sr=16000)
    else:
        raise ValueError("Unsupported file type: only .wav and .mp3 are allowed")


def upload_file():
    """
    Prompts the user to enter a file path to the audio file.
    """
    file_path = input("Enter the path to your audio file (.mp3 or .wav): ").strip()
    return Path(file_path) if file_path else None


# ===================== Duplicate Detection =====================
def check_duplicate(embedding, profile):
    """
    Checks if a given embedding already exists in the profile (i.e., is a duplicate).
    """
    DUPLICATE_THRESHOLD = 0.07
    for emb_list in profile.values():
        for emb in emb_list:
            distance = cosine(np.array(embedding).flatten(), np.array(emb).flatten())
            if distance < DUPLICATE_THRESHOLD:
                return True
    return False


# ===================== Speaker Matching =====================
def match_speaker(new_embedding, profiles):
    """
    Matches the new embedding against all stored profiles using average cosine distance.
    """
    for name, emb_list in profiles.items():
        total_distance = 0
        count = len(emb_list)

        for saved_emb in emb_list:
            distance = cosine(np.array(new_embedding).flatten(), np.array(saved_emb).flatten())
            total_distance += distance

        if count > 0:
            avg = total_distance / count
            if avg < THRESHOLD:
                confidence = 1 - avg 
                confidence = confidence*100
                print(f"✅ SPEAKER IS A MATCH — {name.upper()} — CONFIDENCE: {confidence:.1f}%")
                return name
    return None


# ===================== Profile Management =====================
def add_embedding_to_user(name, new_embedding, profiles):
    """
    Adds a new embedding to a user's profile. Creates new profile if needed.
    """
    if name in profiles:
        profiles[name].append(new_embedding.tolist())
    else:
        profiles[name] = [new_embedding.tolist()]

    with open(PROFILE_PATH, "w") as f:
        json.dump(profiles, f, indent=2)


# ===================== Speaker Recognition =====================
def recognize_speaker(embedding, profiles):
    """
    Recognizes a speaker or prompts to register a new one.
    """
    matched_name = match_speaker(embedding, profiles)
    if matched_name:
        if not check_duplicate(embedding, profiles):
            add_embedding_to_user(matched_name, embedding, profiles)
    else:
        print("❌ Unknown speaker.")
        name = input("Enter the name of the speaker: ").strip()
        add_embedding_to_user(name, embedding, profiles)
        print(f"✅ New speaker '{name}' added.")


# ===================== Voice Comparison =====================
def voice_comparison(profiles, embedding):
    """
    Compares current embedding against a selected profile.
    """
    print("Available Profiles:")
    for i, name in enumerate(profiles.keys(), start=1):
        print(f"{i}. {name}")

    user_name = input("Enter the name of the profile to compare against: ").strip()
    if user_name not in profiles:
        print("❌ Profile not found.")
        return

    name_embeddings = profiles[user_name]
    total_distance = 0
    count = len(name_embeddings)

    for saved_emb in name_embeddings:
        distance = cosine(np.array(saved_emb).flatten(), np.array(embedding).flatten())
        total_distance += distance

    avg = total_distance / count
    confidence = 1 - avg 
    confidence = confidence*100

    if avg < THRESHOLD:
        print(f"✅ SPEAKER IS A MATCH — CONFIDENCE: {confidence:.1f}%")
    else:
        print(f"❌ Speaker is NOT a match. CONFIDENCE: {confidence:.1f}%")


# ===================== Main Script =====================
if __name__ == "__main__":

    # Load or create profiles
    if PROFILE_PATH.exists():
        try:
            with open(PROFILE_PATH, "r") as f:
                profiles = json.load(f)
        except json.JSONDecodeError:
            print("⚠️ Warning: profiles.json is corrupted. Starting with an empty profile.")
            profiles = {}
    else:
        profiles = {}

    encoder = VoiceEncoder()

    while True:
        print("\n==== Speaker Recognition System ====")
        choice = input("Choose an option:\n  1. Identify Speaker\n  2. Compare to Known Speaker\nYour Choice: ").strip()

        audio_path = upload_file()
        if not audio_path:
            print("⚠️ No file provided. Exiting.")
            break

        try:
            wav = load_file(audio_path)
        except Exception as e:
            print(f"❌ Error loading audio: {e}")
            continue

        embedding = encoder.embed_utterance(wav)

        if choice == "1":
            recognize_speaker(embedding, profiles)
        elif choice == "2":
            voice_comparison(profiles, embedding)
        else:
            print("❌ Invalid choice. Try again.")

        cont = input("\nDo you want to process another file? (y/n): ").strip().lower()
        if cont != "y":
            print("👋 Exiting speaker recognition.")
            break
