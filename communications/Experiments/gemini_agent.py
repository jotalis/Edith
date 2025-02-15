import asyncio
import os
import sys
import traceback

import pyaudio
from google import genai
from dotenv import load_dotenv

# Load environment variables from a .env file
load_dotenv()

# Audio configuration
FORMAT = pyaudio.paInt16
CHANNELS = 1
SEND_SAMPLE_RATE = 16000
RECEIVE_SAMPLE_RATE = 24000
CHUNK_SIZE = 1024

# GenAI model configuration
MODEL = "models/gemini-2.0-flash-exp"
CONFIG = {"generation_config": {"response_modalities": ["AUDIO"]}}

# Initialize the GenAI client using the GOOGLE_API_KEY from the .env file.
CLIENT = genai.Client(http_options={"api_version": "v1alpha"}, api_key=os.getenv("GEMINI_API_KEY"))

class AudioChat:
    def __init__(self):
        self.audio_in_queue = asyncio.Queue()        # Queue for incoming audio from the API.
        self.out_queue = asyncio.Queue(maxsize=5)      # Queue for outgoing microphone audio.
        self.session = None
        self.pya = pyaudio.PyAudio()

    async def listen_audio(self):
        """
        Capture audio from the default microphone and enqueue it for sending.
        """
        mic_info = self.pya.get_default_input_device_info()
        audio_stream = await asyncio.to_thread(
            self.pya.open,
            format=FORMAT,
            channels=CHANNELS,
            rate=SEND_SAMPLE_RATE,
            input=True,
            input_device_index=mic_info["index"],
            frames_per_buffer=CHUNK_SIZE,
        )
        # In debug mode, disable exception_on_overflow.
        kwargs = {"exception_on_overflow": False} if __debug__ else {}
        while True:
            data = await asyncio.to_thread(audio_stream.read, CHUNK_SIZE, **kwargs)
            await self.out_queue.put({"data": data, "mime_type": "audio/pcm"})

    async def send_realtime(self):
        """
        Read audio chunks from the output queue and send them to the GenAI session.
        """
        while True:
            msg = await self.out_queue.get()
            await self.session.send(input=msg)

    async def receive_audio(self):
        """
        Receive audio responses from the GenAI session and enqueue them for playback.
        Any text responses (if provided) are printed to the console.
        """
        while True:
            turn = self.session.receive()
            async for response in turn:
                if response.data:
                    self.audio_in_queue.put_nowait(response.data)
                if response.text:
                    print(response.text, end="")
            # Clear any leftover audio (useful if an interruption occurs).
            while not self.audio_in_queue.empty():
                self.audio_in_queue.get_nowait()

    async def play_audio(self):
        """
        Play the received audio via the system's default output.
        """
        stream = await asyncio.to_thread(
            self.pya.open,
            format=FORMAT,
            channels=CHANNELS,
            rate=RECEIVE_SAMPLE_RATE,
            output=True,
        )
        while True:
            bytestream = await self.audio_in_queue.get()
            await asyncio.to_thread(stream.write, bytestream)

    async def run(self):
        """
        Establish the GenAI session and start all asynchronous audio tasks.
        Runs indefinitely until canceled (e.g., via Ctrl+C).
        """
        try:
            async with CLIENT.aio.live.connect(model=MODEL, config=CONFIG) as session, asyncio.TaskGroup() as tg:
                self.session = session
                tg.create_task(self.send_realtime())
                tg.create_task(self.listen_audio())
                tg.create_task(self.receive_audio())
                tg.create_task(self.play_audio())
                await asyncio.Future()  # Run indefinitely.
        except asyncio.CancelledError:
            pass
        except Exception as e:
            traceback.print_exception(e)
        finally:
            self.pya.terminate()


if __name__ == "__main__":
    audio_chat = AudioChat()
    try:
        asyncio.run(audio_chat.run())
    except KeyboardInterrupt:
        print("Exiting on user interrupt.")
