# Voice Age & Emotion Detection System

An advanced audio analysis application designed to estimate a person's age and emotion from voice recordings, incorporating conditional logic for demographic-specific processing.

## 📁 Repository Contents

* **`Age_Emotion_Detection_Voice_working.ipynb`**: The research notebook covering dataset preprocessing, model architecture design, and training.
* **`app.py`**: The production-ready Streamlit application for real-time audio processing and UI-based demographic validation.
* **`requirements.txt`**: A comprehensive list of Python dependencies required for the environment.
* **`packages.txt`**: A list of system-level packages required for specialized library support (such as audio/video codecs).

## 🚀 Features

* **Audio-Based Demographics**: Extracts age and gender features using a custom `Wav2Vec2` architecture.
* **Conditional Workflow**:
* **Gender Filtering**: Strictly accepts male voices; rejects female voices.
* **Senior Detection**: Automatically triggers secondary emotion analysis if the estimated age is greater than 60.


* **Robust Pipeline**: Implements `InferenceWrapper` for multi-task classification.

## 🔧 Setup & Installation

### 1. Prerequisites

Clone this repository:

```bash
git clone https://github.com/KVAlwaysLearning/Age_Emotion_Detection_with_voice_Sub
cd Age_Emotion_Detection_with_voice_Sub

```

### 2. Install Dependencies

Install all required libraries and system packages:

```bash
pip install -r requirements.txt
# If deploying to Linux-based environments (like Streamlit Cloud):
sudo apt-get install -y $(cat packages.txt)

```

**Key Packages:**

* `streamlit`: The interactive web interface.
* `transformers` & `torch`: Wav2Vec2 inference.
* `librosa`: Audio signal processing.
* `ffmpeg`/`libasound2`: System-level dependencies listed in `packages.txt` for handling raw audio/video streams.

### 3. Model Initialization

The application downloads pre-trained model weights into a `./Models/` directory upon the first execution. Ensure your environment has write permissions.

## 💻 Usage

### Running the App

Launch the web interface locally:

```bash
streamlit run app.py

```

### Exploring the Research

You can open `Age_Emotion_Detection_Voice_working.ipynb` in any Jupyter environment to inspect the model training logic and verification results.

## 📂 Project Structure

```text
├── Models/            # Directory for downloaded model weights or custom trained weights (Refer .ipynb file)
├── app.py             # Streamlit web application
├── Age_Emotion_Detection_Voice_working.ipynb # Research and training notebook
├── requirements.txt   # Python dependencies
├── packages.txt       # System-level dependencies
└── README.md          # Project documentation

```

## 🔗 Links

* **Live App**: [Voice Analysis App](https://ageemotiondetectionwithvoice-app.streamlit.app/)
* **GitHub Repo**: [Age & Emotion Detection Repository](https://github.com/KVAlwaysLearning/Age_Emotion_Detection_with_voice_Sub)

---

Visuals:

<img width="738" height="273" alt="App_1" src="https://github.com/user-attachments/assets/f2d4f658-8f7e-4377-af56-483b16057499" />

