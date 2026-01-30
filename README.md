
# Vocal Agent - Stutter2Fluent

A real-time, AI-powered speech therapy assistant designed to help users improve fluency. This application uses advanced speech-to-speech technology to analyze stuttering patterns, provide therapeutic feedback, and generate fluent audio/video output.

## 🌟 Features

- **Real-Time Streaming**: Low-latency speech analysis using WebSockets and AudioWorklet.
- **Multilingual Support**: Supports English, Hindi, Telugu, Spanish, French, and many more.
- **AI Analysis**: Uses **Ollama (Llama 3.1)** and **Agno** to detect stuttering types (Repetitions, Blocks, Prolongations) and provide clinical SOAP notes.
- **Fluent Regeneration**: Reconstructs fragmented speech into fluent text and audio using **Kokoro TTS** (High Quality) or **Edge TTS**.
- **Video Dubbing**: Merges the generated fluent audio back into the original video, preserving lip-sync where possible.
- **Acoustic Features**: Analyzes RMS Energy and Zero Crossing Rate to detect physical tension and struggle behaviors.

## 🛠️ Prerequisites

Before running the application, ensure you have the following installed:

1. **Python 3.10, 3.11, or 3.12** (Python 3.13+ is currently incompatible with `faster-whisper`).
2. **FFmpeg**: Required for video processing.
    - Windows: `winget install Gyan.FFmpeg`
    - Mac: `brew install ffmpeg`
    - Linux: `sudo apt install ffmpeg`
3. **Ollama**: Required for the AI Agent logic.
    - Download from ollama.com.
    - Pull the model: `ollama pull llama3.1:8b`

## 🚀 Installation & Setup

1. **Clone the Repository**

    ```bash
    git clone <your-repo-url>
    cd Stutter2Fluent
    ```

2. **Create a Virtual Environment** (Recommended)

    ```bash
    python -m venv venv
    # Windows
    venv\Scripts\activate
    # Mac/Linux
    source venv/bin/activate
    ```

3. **Run the Setup Script**
    This script installs Python dependencies and downloads necessary AI models (Kokoro ONNX).

    ```bash
    python download_requirements.py
    ```

4. **Start Ollama**
    Ensure Ollama is running in the background.

    ```bash
    ollama serve
    ```

## ▶️ Running the Application

Start the FastAPI server:

```bash
python main.py
```

- The application will launch at `http://localhost:8000`.
- If port 8000 is busy, you can change it by setting the `PORT` environment variable.

## 📂 Project Structure

- `main.py`: Core backend logic (FastAPI, WebSocket, Audio Pipeline).
- `index.html`: Frontend UI (Bento Grid design, Audio Recording, Streaming).
- `download_requirements.py`: Helper script for setup.
- `requirements.txt`: Python dependency list.
- `temp/`: Stores temporary audio/video files during processing (auto-cleaned).

## ⚙️ Configuration

You can configure the application using environment variables or a `.env` file:

- `OLLAMA_HOST`: URL of the Ollama server (default: `http://127.0.0.1:11434`).
- `PORT`: Port to run the web server on (default: `8000`).

## 🧠 How It Works

1. **Input**: User records audio/video or streams live via WebSocket.
2. **Transcription**: **Faster-Whisper** converts speech to text.
3. **Acoustic Analysis**: System calculates Energy and Zero Crossing Rate to detect non-verbal struggle.
4. **Agent Reasoning**: The AI (Llama 3.1) analyzes the text + acoustic features to identify dysfluencies and determine a therapy strategy (e.g., "Fluency Shaping").
5. **Synthesis**: **Kokoro TTS** or **Edge TTS** generates a fluent version of the speech.
6. **Output**: The user receives the fluent audio, a dubbed video, and a detailed clinical analysis.

## 🤝 Contributing

1. Fork the repository.
2. Create a feature branch (`git checkout -b feature/NewFeature`).
3. Commit your changes.
4. Push to the branch.
5. Open a Pull Request.

## 📄 License

MIT License

```

<!--
[PROMPT_SUGGESTION]How can I create a Dockerfile to containerize this application for easier deployment?[/PROMPT_SUGGESTION]
[PROMPT_SUGGESTION]Explain how to set up a GitHub Action to automatically lint the Python code on push.[/PROMPT_SUGGESTION]
