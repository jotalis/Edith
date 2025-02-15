import asyncio
import os
import traceback
import json
import time

import pyaudio
import cv2
import numpy as np
from google import genai
from dotenv import load_dotenv

import websockets  # For both server and GPU client connections

from google.genai.types import (
    FunctionDeclaration,
    GoogleSearch,
    LiveConnectConfig,
    PrebuiltVoiceConfig,
    SpeechConfig,
    Tool,
    ToolCodeExecution,
    VoiceConfig,
)

# Load environment variables from a .env file
load_dotenv()

# Audio configuration
FORMAT = pyaudio.paInt16
CHANNELS = 1
SEND_SAMPLE_RATE = 16000
RECEIVE_SAMPLE_RATE = 24000
CHUNK_SIZE = 1024

find_object = FunctionDeclaration(
    name="find_object",
    description=(
        "Find an object by retrieving the details and location of a specified object. "
        "This function should be called when the user asks about the whereabouts of an object, "
        "such as 'Help me find my keys' or 'Where is my laptop?'."
    ),
    parameters={
        "type": "OBJECT",
        "properties": {
            "object_name": {
                "type": "STRING",
                "description": "The name of the object to be located.",
            },
        },
        "required": ["object_name"],
    },
)

get_current_weather = FunctionDeclaration(
    name="get_current_weather",
    description="Get current weather in the given location",
    parameters={
        "type": "OBJECT",
        "properties": {
            "location": {
                "type": "STRING",
            },
        },
    },
)

# GenAI model configuration
MODEL = "models/gemini-2.0-flash-exp"

CONFIG = LiveConnectConfig(
    response_modalities=["AUDIO"],
    tools=[Tool(function_declarations=[get_current_weather, find_object])],
    speech_config=SpeechConfig(
        voice_config=VoiceConfig(
            prebuilt_voice_config=PrebuiltVoiceConfig(
                voice_name="Aoede",
            )
        )
    ),
)

# Initialize the GenAI client using the GEMINI_API_KEY from the .env file.
CLIENT = genai.Client(
    http_options={"api_version": "v1alpha"}, api_key=os.getenv("GEMINI_API_KEY")
)


class AudioChat:
    def __init__(self):
        self.audio_in_queue = asyncio.Queue()  # Queue for incoming audio from the API.
        self.out_queue = asyncio.Queue(maxsize=5)  # Queue for outgoing microphone audio.
        self.session = None
        self.pya = pyaudio.PyAudio()
        # Event to control whether audio from the mic should be sent (disabled by default)
        self.audio_enabled_event = asyncio.Event()
        # Set to keep track of connected websocket clients (microcontrollers)
        self.ws_connections = set()
        # Persistent connection to the GPU server (initially None)
        self.gpu_ws = None

    async def listen_audio(self):
        """
        Capture audio from the default microphone and enqueue it for sending.
        Audio is captured only if the audio_enabled_event is set (e.g. after a button press).
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
        kwargs = {"exception_on_overflow": False} if __debug__ else {}
        while True:
            if self.audio_enabled_event.is_set():
                data = await asyncio.to_thread(audio_stream.read, CHUNK_SIZE, **kwargs)
                await self.out_queue.put({"data": data, "mime_type": "audio/pcm"})
            else:
                await asyncio.sleep(0.1)

    async def send_realtime(self):
        """
        Read audio chunks from the output queue and send them to the Gemini session.
        """
        while True:
            msg = await self.out_queue.get()
            await self.session.send(input=msg)

    async def get_object(self, object_name: str) -> str:
        """
        Dummy implementation for a function call from Gemini.
        Broadcasts a "VIBRATE" message to all connected microcontrollers,
        then (after receiving image data from the microcontroller) the images
        will be processed and sent to the GPU server.
        """
        if self.ws_connections:
            for ws in self.ws_connections.copy():
                try:
                    await ws.send("VIBRATE")
                except Exception as e:
                    print("Error sending VIBRATE message:", e)
        result = f"Object '{object_name}' located at the default coordinates (42.3601, -71.0589)."
        return result

    async def send_image_to_gpu(self, image_data: bytes):
        """
        Process an image received from the microcontroller:
          - Decode the received image.
          - Resize it to 480p (854x480).
          - Compress it using JPEG encoding with a quality level of 60.
          - Send the processed image bytes over the persistent GPU websocket.
        """
        np_arr = np.frombuffer(image_data, np.uint8)
        frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
        if frame is None:
            print("Failed to decode received image.")
            return

        frame = cv2.resize(frame, (854, 480))
        encode_params = [int(cv2.IMWRITE_JPEG_QUALITY), 60]
        success, buffer = cv2.imencode('.jpg', frame, encode_params)
        if not success:
            print("Failed to encode image as JPEG.")
            return
        frame_bytes = buffer.tobytes()

        if self.gpu_ws is None:
            print("GPU websocket not connected, cannot send image.")
            return
        try:
            await self.gpu_ws.send(frame_bytes)
            print("Sent processed image to GPU server.")
        except Exception as e:
            print("Error sending image to GPU server:", e)

    async def receive_audio(self):
        """
        Receive audio responses from the Gemini session and enqueue them for playback.
        Additionally, if the response includes a function call, handle it.
        """
        while True:
            turn = self.session.receive()
            async for response in turn:
                if response.tool_call:
                    for function_call in response.tool_call.function_calls:
                        print("function_call", function_call)
                        if function_call.name == "find_object":
                            print(function_call.args["object_name"])
                            result = await self.get_object(function_call.args["object_name"])
                            print(result)
                            await self.session.send(input=result, end_of_turn=True)
                if response.data:
                    self.audio_in_queue.put_nowait(response.data)
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

    async def websocket_handler(self, websocket, path=None):
        """
        Handle incoming WebSocket connections from the microcontroller.
        Expect messages like "BUTTON_PRESSED", "BUTTON_RELEASED", or binary image data.
        When binary data (an image) is received, process and forward it to the GPU server.
        """
        self.ws_connections.add(websocket)
        try:
            async for message in websocket:
                if isinstance(message, bytes):
                    await self.send_image_to_gpu(message)
                else:
                    print("Received message from microcontroller:", message)
                    if message == "BUTTON_PRESSED":
                        print("Enabling audio transmission.")
                        self.audio_enabled_event.set()
                    elif message == "BUTTON_RELEASED":
                        print("Disabling audio transmission.")
                        self.audio_enabled_event.clear()
                    else:
                        print("Unknown command received:", message)
        finally:
            self.ws_connections.remove(websocket)

    async def run_websocket_server(self):
        """
        Run a WebSocket server to listen for commands and image data from the microcontroller.
        """
        async with websockets.serve(self.websocket_handler, "0.0.0.0", 8765):
            print("WebSocket server started on ws://0.0.0.0:8765")
            await asyncio.Future()  # run forever

    async def connect_to_gpu(self):
        """
        Establish and maintain a persistent connection to the GPU websocket.
        If the connection is lost, this task will attempt to reconnect.
        """
        while True:
            try:
                self.gpu_ws = await websockets.connect("ws://172.24.75.90:8000")
                print("Connected to GPU websocket.")
                await self.gpu_ws.wait_closed()
                print("GPU websocket connection closed. Reconnecting...")
            except Exception as e:
                print("Error connecting to GPU websocket:", e)
            self.gpu_ws = None
            await asyncio.sleep(5)

    async def run(self):
        """
        Establish the Gemini session, start audio tasks, and run the WebSocket server concurrently.
        Also starts the persistent GPU websocket connection.
        """
        try:
            async with CLIENT.aio.live.connect(model=MODEL, config=CONFIG) as session, asyncio.TaskGroup() as tg:
                self.session = session
                tg.create_task(self.send_realtime())
                tg.create_task(self.listen_audio())
                tg.create_task(self.receive_audio())
                tg.create_task(self.play_audio())
                tg.create_task(self.run_websocket_server())
                tg.create_task(self.connect_to_gpu())
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
