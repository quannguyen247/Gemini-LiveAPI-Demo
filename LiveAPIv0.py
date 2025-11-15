import asyncio
import os
from google import genai
from google.genai.types import LiveConnectConfig, Modality, Content, Part
from dotenv import load_dotenv

load_dotenv()

try:
    client = genai.Client(api_key=os.environ["GEMINI_KEY"])
except KeyError:
    print("Error: Please create a .env file and add your GEMINI_KEY.")
    exit()

model = "gemini-2.0-flash-live-001" # default model

config = LiveConnectConfig(
    response_modalities=[Modality.TEXT],
    system_instruction="You are a helpful assistant and answer in a friendly tone.",
)

async def main():
    try:
        text_input = input("Enter your text: ")
        
        async with client.aio.live.connect(model=model, config=config) as session:
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
        print(f"Error during interaction: {e}")

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\nProgram stopped.")