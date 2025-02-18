# Personal Yoga Coach (On-Device AI)

Personal Yoga Coach is a cutting-edge, AI-powered application designed to provide personalized yoga guidance directly on your device. The app enhances your yoga practice by offering real-time feedback, customized routines, and on-demand coaching—all while operating entirely offline to ensure privacy and accessibility.

This project won the **Meta Consumer AI on the Edge Hackathon in December 2024**. Huge thanks to the **Lightning-Fast Teammates** for making this possible: **Gaspard Hassenforder, Alex Bohane, Lazlo Guioland, Julien Roubaud, and Alba Maria Tellez Fernandez**.

---

## 🌟 Features

- **🖥️ On-Device AI**: Runs offline using Qwen/Qwen2.5-0.5B-Instruct for local LLM inference, ensuring privacy and no dependency on the internet.
- **🧘 Real-Time Feedback**: AI-powered pose correction and accuracy scoring to help improve posture and technique.
- **🗣️ Voice and Visual Guidance**: Provides both verbal cues and on-screen visual overlays for pose corrections.
- **🦾 Computer Vision-Powered Analysis**: Uses **MediaPipe** and **OpenCV** for real-time human pose detection and skeleton tracking.
- **🎙️ AI-Driven Voice Feedback**: NLP-driven corrections via **Qwen LLM**, using structured prompt engineering for precise feedback. Integrated with **Piper TTS** for voice-based corrections.
- **📊 Adaptive Feedback Mechanism**: Scores pose accuracy and dynamically adjusts the difficulty level and feedback precision based on user performance.
- **🖥️ Gradio UI for User Interaction**: Browser-based interactive interface that provides real-time visualization of detected poses and feedback.
- **⚡ Optimized Performance**: Implements multi-threaded pose analysis, caching mechanisms, and lightweight inference models for seamless execution on local devices.

---

## 🔧 How It Works

1. **Pose Detection**:
   - Utilizes **MediaPipe's BlazePose** for extracting human body key points.
   - Processes the skeletal structure using **OpenCV** overlays.
   
2. **AI-Powered Correction**:
   - Compares detected poses with predefined yoga posture templates.
   - Uses **Qwen LLM** to provide textual and audio-based corrections.
   
3. **Real-Time Feedback**:
   - Processes corrections dynamically and offers **voice guidance via Piper TTS**.
   - Displays **visual cues on Gradio UI** for better understanding.
   
4. **Adaptive Accuracy Scoring**:
   - Computes pose similarity scores using cosine similarity between detected and ideal skeletal points.
   - Adjusts guidance based on user performance over multiple sessions.

---

## 🛠️ Installation Guide

Follow these steps to install and run Personal Yoga Coach on your local machine.

### Prerequisites
- Python 3.8 or higher
- pip
- Virtual environment support (optional but recommended)

### Installation Steps

1. **Clone the Repository**:
    ```bash
    git clone https://github.com/LazGL/YogaLivePoseCorrector.git
    ```

2. **Navigate to the Project Directory**:
    ```bash
    cd YogaLivePoseCorrector
    ```

3. **Create a Virtual Environment** (Recommended):
    ```bash
    python -m venv venv
    ```

4. **Activate the Virtual Environment**:
    - On Windows:
        ```bash
        .\venv\Scripts\activate
        ```
    - On macOS/Linux:
        ```bash
        source venv/bin/activate
        ```

5. **Install Dependencies**:
    ```bash
    pip install -r requirements.txt
    ```

6. **Run the Application**:
    ```bash
    python app_llm.py
    ```

---

## 🚀 Usage

1. Launch the application following the installation steps.
2. Open the Gradio UI in your browser.
3. Follow the on-screen instructions to start your yoga session.
4. Receive real-time feedback through voice and visual guidance.
5. Adjust your posture based on the corrections provided by the AI.

---

## 🛡️ Privacy & Security

- All processing occurs **entirely on-device**, ensuring that your data never leaves your system.
- No internet connection is required, providing a **secure and private experience**.
- Your movement data and corrections remain completely confidential.

---

## 🤝 Contributing

We welcome contributions to enhance Personal Yoga Coach! If you'd like to contribute:
1. Fork the repository.
2. Create a new branch (`git checkout -b feature-branch`).
3. Commit your changes (`git commit -m "Add new feature"`).
4. Push to the branch (`git push origin feature-branch`).
5. Open a pull request.

---

## 📜 License

This project is licensed under the cc-by-nc License. See the [LICENSE](LICENSE) file for more details.

---

## 📞 Support

For any issues or suggestions, feel free to open an issue on GitHub or contact us via email.

🔗 **GitHub Repository**: [YogaLivePoseCorrector](https://github.com/LazGL/YogaLivePoseCorrector)

