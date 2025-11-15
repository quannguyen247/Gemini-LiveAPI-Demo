import asyncio
import os
import io
import wave
from google import genai
from google.genai.types import LiveConnectConfig, Modality, Content, Part, Blob
from dotenv import load_dotenv
import soundfile as sf
import librosa

load_dotenv()

try:
    client = genai.Client(api_key=os.environ["GEMINI_KEY"])
except KeyError:
    print("Error: Please create a .env file and add your GEMINI_KEY.")
    exit()

model_text = "gemini-2.0-flash-live-001"
model_audio = "gemini-2.5-flash-native-audio-preview-09-2025"
model_audio_timeout = 10.0  # seconds
audio_input_sr = 16000  # sample rate for audio recording
audio_output_sr = 24000  # sample rate for audio playback
output_audio_file = "response_audio.wav"  # output file name for audio response

config_text = LiveConnectConfig(
    response_modalities=[Modality.TEXT],
    system_instruction="You are a helpful assistant and answer in a friendly tone.",
)

# Configuration for audio-in, audio-out
config_audio = LiveConnectConfig(
    response_modalities=[Modality.AUDIO],
    system_instruction="You are a helpful assistant and answer in a friendly tone.",
)

async def main():
    print("Choose input type:")
    print("1: Audio")
    print("2: Text")
    choice = input("Enter your choice (1 or 2): ")

    if choice == "1":
        try:
            audio_path = input("Enter the path to the audio file: ")
            if not os.path.exists(audio_path):
                print("Error: Audio file does not exist.")
                return

            async with client.aio.live.connect(model=model_audio, config=config_audio) as session:
                print("Processing audio...")
                # Use librosa to load and convert audio to 16kHz
                y, sr = librosa.load(audio_path, sr=16000)

                # Write to buffer as PCM 16-bit
                buffer = io.BytesIO()
                sf.write(buffer, y, sr, format='RAW', subtype='PCM_16')
                buffer.seek(0)
                audio_bytes = buffer.read()

                # Send processed audio using send_realtime_input
                await session.send_realtime_input(
                    audio=Blob(mime_type="audio/pcm;rate=16000", data=audio_bytes)
                )

                print("Receiving audio response and saving to file...")
                
                # Collect audio chunks with timeout
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
                    
                    # Stage 2: Once response started, read all remaining data
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
                    print(f"⚠️  Model did not provide any response within {model_audio_timeout} seconds.")
                    timed_out = True
                
                # Write to file if audio data was received
                if audio_chunks and not timed_out:
                    with wave.open(output_audio_file, "wb") as wf:
                        wf.setnchannels(1)
                        wf.setsampwidth(2)
                        wf.setframerate(audio_output_sr)
                        wf.writeframes(b"".join(audio_chunks))
                    print(f"✔ Audio response saved to file: {output_audio_file}")
                elif not audio_chunks and not timed_out:
                    print("ℹ️  No audio data was received in the response.")
        except Exception as e:
            print(f"Error during audio processing: {e}")

    elif choice == "2":
        try:
            text_input = input("Enter text content: ")

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

    else:
        print("Invalid choice. Please run the program again.")

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\nProgram stopped.")
    except Exception as e:
        print(f"Unexpected error: {e}")