# Gemini Live API: Real-time Voice Chat Examples

This repository provides a collection of Python examples demonstrating the capabilities of the Google Gemini Live API. The project showcases a progressive evolution from a simple text-based chat to a full-featured, real-time, two-way voice conversation, illustrating how to implement and scale features with the API.

## Features (LiveAPIv3.py)

-   **Real-time Two-Way Audio:** Engage in a live, low-latency voice conversation with the Gemini model.
-   **Microphone Input:** Captures audio directly from your microphone for seamless interaction.
-   **Live Audio Playback:** Plays the model's voice response directly to your speakers without saving to a file.
-   **Text-Based Chat:** Includes a separate mode for traditional text-based interaction.
-   **System Configuration:** Allows for microphone testing and command-line theme customization for a better user experience.
-   **Robust Asynchronous Handling:** Built with `asyncio` to manage concurrent tasks like audio recording, streaming, and receiving data.

## Project Evolution

The repository contains several script versions, each building upon the last. This structure is designed to provide a clear learning path.

| Version | Goal | Interaction Type | Audio Input | Audio Output | Key Libraries |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **`LiveAPIv0.py`** | Basic API Demo | Text-only | ❌ None | ❌ None | `google-genai` |
| **`LiveAPIv1.py`** | File-based Audio | Text or Audio | From audio file | Writes to `.wav` file | `librosa`, `soundfile` |
| **`LiveAPIv2.py`** | Real-time Audio (Basic) | Text or Audio | 🎙️ **Microphone** | Writes to `.wav` file | `sounddevice` |
| **`LiveAPIv3.py`** | Real-time Voice Chat | Text or Audio | 🎙️ **Microphone** | 🔊 **Live Speaker Playback** | `sounddevice`, `numpy` |

## Prerequisites

-   Python 3.9+
-   A Google Gemini API Key.

## Installation

1.  **Clone the repository:**
    ```bash
    git clone <your-repository-url>
    cd <your-repository-directory>
    ```

2.  **Install the required Python libraries:**
    ```bash
    pip install google-genai python-dotenv sounddevice soundfile librosa numpy
    ```
    *Note: On some systems, you may need to install system-level audio libraries like `portaudio` for `sounddevice` to work correctly.*

## Configuration

1.  Create a file named `.env` in the root of the project directory.
2.  Add your Gemini API key to the `.env` file as follows:

    ```
    GEMINI_KEY="YOUR_API_KEY_HERE"
    ```

## Usage

The main, most feature-complete script is `LiveAPIv3.py`.

1.  **Run the application:**
    ```bash
    python LiveAPIv3.py
    ```

2.  **Select an option from the menu:**
    -   **`1. Real-time Audio Interaction`**: Start a voice conversation. Press `Enter` to stop recording your voice and wait for the model's response.
    -   **`2. Text Interaction`**: Start a text-based chat session.
    -   **`3. Test Microphone`**: Record and play back a short audio clip to verify your microphone is working.
    -   **`4. Config Theme`**: Change the color scheme of the command-line interface.
    -   **`5. Exit`**: Close the program.

The other scripts (`LiveAPIv0.py`, `LiveAPIv1.py`, `LiveAPIv2.py`) can be run similarly and are provided for educational purposes to understand the development progression.

## License

This project is licensed under the MIT License. See the `LICENSE` file for details.
