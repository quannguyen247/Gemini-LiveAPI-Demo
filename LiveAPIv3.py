import asyncio
import os
import numpy as np
from google import genai
from google.genai.types import LiveConnectConfig, Modality, Content, Part, Blob
from dotenv import load_dotenv
import sounddevice as sd

# ==================================================================
#                       GLOBAL CONFIGURATION
# ==================================================================

model_global = None # Placeholder for global model variable
model_text = "gemini-2.0-flash-live-001"
model_audio = "gemini-2.5-flash-native-audio-preview-09-2025"
model_audio_timeout = 10.0 # seconds
audio_input_sr = 16000 # sample rate for audio recording
audio_output_sr = 24000 # sample rate for audio playback
mic_test_duration = 5.0 # seconds for microphone test

# Load environment variables
load_dotenv()

try:
    client = genai.Client(api_key=os.environ["GEMINI_KEY"])
except KeyError:
    print("Error: Please create a .env file and add your GEMINI_KEY.")
    exit()

config_text = LiveConnectConfig(
    response_modalities=[Modality.TEXT],
    system_instruction="You are a helpful assistant and answer in a friendly tone.",
)

config_audio = LiveConnectConfig(
    response_modalities=[Modality.AUDIO],
    system_instruction="You are a helpful assistant and answer in a friendly tone.",
)

# ==================================================================
#                       TEXT INTERACTION
# ==================================================================

async def text_interaction():
    """Handles text-based interaction."""
    print("\n💬 Starting text interaction...")
    print(">> Type 'quit' or 'exit' to return to the main menu.")
    while True:
        try:
            text_input = input("You: ")
            if text_input.lower() in ["quit", "exit"]:
                break

            async with client.aio.live.connect(model=model_text, config=config_text) as session:
                await session.send_client_content(
                    turns=Content(role="user", parts=[Part(text=text_input)]),
                    turn_complete=True
                )

                print("Model: ", end="", flush=True)
                async for chunk in session.receive():
                    if chunk.server_content and chunk.server_content.model_turn:
                        for part in chunk.server_content.model_turn.parts:
                            if part.text:
                                print(part.text, end="", flush=True)
                print()
        except Exception as e:
            print(f"Error during text interaction: {e}")

# ==================================================================
#                       MICROPHONE TEST
# ==================================================================

def test_microphone():
    """Records a short audio clip to test the microphone."""
    try:
        print(f"Starting recording for {mic_test_duration} seconds...")
        recording = sd.rec(int(mic_test_duration * audio_input_sr),
                           samplerate=audio_input_sr,
                           channels=1,
                           dtype='float32')
        sd.wait()

        print("Recording finished. Playing back for verification...")
        sd.play(recording, audio_input_sr)
        sd.wait()

        print("✔ Microphone is working correctly.")
    except Exception as e:
        print(f"Microphone test error: {e}")

# ==================================================================
#                       THEME CONFIGURATION
# ==================================================================

def config_theme():
    """Allows the user to select a CMD theme."""
    themes = {
        "0": {"name": "Default (White on Black)", "code": "07"},
        "1": {"name": "Hacker (Green on Black)", "code": "02"},
        "2": {"name": "Light (Black on White)", "code": "F0"},
        "3": {"name": "Blue (White on Blue)", "code": "1F"},
    }

    while True:
        print("\nTheme Selection: ")
        for key, theme in themes.items():
            print(f"{key}. {theme['name']}")
        
        choice = input("Select a theme option: ")

        if choice in themes:
            os.system(f"color {themes[choice]['code']}")
            print(f"🎨 Theme changed to: {themes[choice]['name']}")
            break
        else:
            print("Invalid option! Please try again.")

# ==================================================================
#              REAL-TIME AUDIO INTERACTION
# ==================================================================

async def real_time_audio_interaction():
    async with client.aio.live.connect(model=model_audio, config=config_audio) as session:
        print("\n🎤 Starting real-time audio session...")
        print(">> Type 'quit' or 'exit' to end the session.")

        while True:
            queue: asyncio.Queue[np.ndarray] = asyncio.Queue()

            # Task to wait for user to press Enter or type quit/exit
            user_input_task = asyncio.create_task(
                asyncio.to_thread(
                    input,
                    ">> Press Enter to stop recording, type 'quit' or 'exit' to end the session.\n"
                )
            )

            loop = asyncio.get_running_loop()

            # The sounddevice callback runs in a separate thread
            def audio_callback(indata, frames, time, status):
                if status:
                    print(status)
                loop.call_soon_threadsafe(queue.put_nowait, indata.copy())

            # Task to send audio from the queue to the API
            async def sender_task():
                while True:
                    try:
                        indata = await asyncio.wait_for(queue.get(), timeout=0.1)
                        if indata is None:
                            break
                        await session.send_realtime_input(
                            audio=Blob(
                                mime_type=f"audio/pcm;rate={audio_input_sr}",
                                data=indata.tobytes()
                            )
                        )
                    except asyncio.TimeoutError:
                        # If user pressed Enter, stop sending
                        if user_input_task.done():
                            break

            sender = asyncio.create_task(sender_task())

            # Start recording from the microphone
            with sd.InputStream(
                samplerate=audio_input_sr,
                channels=1,
                dtype='int16',
                callback=audio_callback
            ):
                user_input = await user_input_task  # Wait for Enter / quit / exit

            if user_input.lower() in ["quit", "exit"]:
                print("Ending audio session.")
                # Stop sender and exit loop
                await queue.put(None)
                await sender
                break

            # Stop the sender_task and wait for it to finish
            await queue.put(None)
            await sender

            print("⏳ Recording stopped. Receiving response from model...")

            # Collect audio chunks from response
            audio_chunks = []
            timed_out = False

            try:
                receiver = session.receive().__aiter__()

                # Stage 1: Timeout only if NO response received
                first_chunk = await asyncio.wait_for(
                    receiver.__anext__(),
                    timeout=model_audio_timeout
                )
                if first_chunk.server_content and first_chunk.server_content.model_turn:
                    for part in first_chunk.server_content.model_turn.parts:
                        if part.inline_data:
                            audio_chunks.append(part.inline_data.data)

                # Stage 2: Once response started, read all remaining data without timeout
                while True:
                    try:
                        chunk = await receiver.__anext__()
                    except StopAsyncIteration:
                        break
                    if chunk.server_content and chunk.server_content.model_turn:
                        for part in chunk.server_content.model_turn.parts:
                            if part.inline_data:
                                audio_chunks.append(part.inline_data.data)

            except asyncio.TimeoutError:
                # No response at all within timeout
                print(f"⚠️  Model did not provide any response within {model_audio_timeout} seconds.")
                timed_out = True

            # Play audio directly to speakers
            if audio_chunks and not timed_out:
                print("✔ Audio response received. Playing now...")
                audio_data = b"".join(audio_chunks)
                audio_array = np.frombuffer(audio_data, dtype=np.int16)
                sd.play(audio_array, samplerate=audio_output_sr)
                sd.wait()
                print("✔ Playback finished.")
            elif not audio_chunks and not timed_out:
                print("ℹ️  No audio data was received in the response.")

# ==================================================================
#                       MAIN MENU
# ==================================================================

async def main():
    while True:
        print("\n=== LiveAPIv3 Interaction Menu ===\n")
        print("1. Real-time Audio Interaction")
        print("2. Text Interaction")
        print("3. Test Microphone")
        print("4. Config Theme")
        print("5. Exit")
        choice = input("Select an option: ")

        match choice:
            case "1":
                await real_time_audio_interaction()
            case "2":
                await text_interaction()
            case "3":
                test_microphone()
            case "4":
                config_theme()
            case "5":
                print("Goodbye!")
                break
            case _:
                print("Invalid option!")

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\nProgram stopped.")