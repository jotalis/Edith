import base64
import asyncio
import os
import traceback

import numpy as np
import cv2
import pyaudio
from google import genai
from dotenv import load_dotenv

import requests
import time

from google.genai.types import (
    FunctionDeclaration,
    LiveConnectConfig,
    PrebuiltVoiceConfig,
    SpeechConfig,
    Tool,
    Content,
    Part,
    VoiceConfig,
)

# Load environment variables from a .env file
load_dotenv()

find_entity = FunctionDeclaration(
    name="find_entity",
    description=(
        "This function should be called when the user asks about the whereabouts or location or finding anything, "
        "such as 'Help me find my keys' or 'Where is my laptop?' or 'help me find the old grandma'."
    ),
    parameters={
        "type": "OBJECT",
        "properties": {
            "object_description": {
                "type": "STRING",
                "description": "A description of the object to be located.",
            },
        },
        "required": ["object_description"],
    },
)

# Audio configuration
FORMAT = pyaudio.paInt16
CHANNELS = 1
SEND_SAMPLE_RATE = 16000
RECEIVE_SAMPLE_RATE = 24000
CHUNK_SIZE = 1024

# GenAI model configuration
MODEL = "models/gemini-2.0-flash-exp"
CONFIG = LiveConnectConfig(
    response_modalities=["AUDIO"],
    tools=[Tool(function_declarations=[find_entity])],
    speech_config=SpeechConfig(
        voice_config=VoiceConfig(
            prebuilt_voice_config=PrebuiltVoiceConfig(
                voice_name="Aoede",
            )
        )
    ),
    system_instruction=Content(
        role="model",
        parts=[
            Part(
                text="""
                You are a companion for the visually impaired which gives them newfound confidence to navigate the world. In other words, you are their sixth sense that uses advanced depth data and 3D spatial reconstruction.

                You have the following tool available to you:
                - find_entity: Find the location of an object in the environment.

                Rules:
                - Whenever you're asked about the location or position of something you can use the find_entity tool. Then, once you have provided information from that tool, give relevant information about how they might reach that object. Are there any paths? Is there something in the way that they need to navigate around? Are there any helpful distinctive features nearby to anchor their understanding of the scene? You should mention things in this vein.
                - If you ask for a description, remember that the user can not see. Use contextual information from images and past function call responses, if possible, to ask guiding questions.
                    """
            )
        ],
    ),
)

# PI Config
PI_IP = "10.32.78.132"
PI_IP = "100.64.146.118"
# PI_IP = "10.32.72.225"
SNAPSHOT_URL = f"http://{PI_IP}:5000/snapshot"
ANGLE_URL = f"http://{PI_IP}:5000/angle"

# GPU

GPU_IP = "172.24.75.90"
GPU_PROCESS_URL = f"http://{GPU_IP}:8404/process"  # Replace with your GPU server URL


# Initialize the GenAI client using the GOOGLE_API_KEY from the .env file.
CLIENT = genai.Client(
    http_options={"api_version": "v1alpha"}, api_key=os.getenv("GEMINI_API_KEY")
)

WAIT = True


class AudioChat:
    def __init__(self):
        self.audio_in_queue = asyncio.Queue()  # Queue for incoming audio from the API.
        self.out_queue = asyncio.Queue(
            maxsize=5
        )  # Queue for outgoing microphone audio.
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
        last_image_time = time.monotonic() - 8

        while True:
            # Send the next audio chunk
            msg = await self.out_queue.get()
            await self.session.send(input=msg)

            now = time.monotonic()
            if now - last_image_time >= 8:
                last_image_time = now  # Update the timestamp
                try:
                    # Fetch the snapshot using requests in a thread
                    response = await asyncio.to_thread(
                        requests.get, SNAPSHOT_URL, timeout=5
                    )
                    if response.status_code == 200:
                        # Here we assume the snapshot returns JPEG image data.
                        image_data = response.content  # Raw binary image data
                        # Prepare the input in the format expected by Gemini live multimodal.
                        # The input must be a dict with keys "data" and "mime_type".
                        image_input = {"data": image_data, "mime_type": "image/jpeg"}

                        # Send the image input to the session
                        await self.session.send(input=image_input)
                    else:
                        print(f"Failed to get snapshot: {response.status_code}")
                except Exception as e:
                    print("Error fetching snapshot:", e)

    async def receive_audio(self):
        """
        Receive audio responses from the GenAI session and enqueue them for playback.
        Any text responses (if provided) are printed to the console.
        """
        while True:
            turn = self.session.receive()
            async for response in turn:
                if response.tool_call:
                    for function_call in response.tool_call.function_calls:
                        print("function_call", function_call)
                        if function_call.name == "find_entity":
                            print(function_call.args["object_description"])
                            result = await self.find_entity(
                                function_call.args["object_description"]
                            )
                            print(result)
                            decoded_str, depth, angle = result
                            # decoded_str = base64.b64decode(image).decode("utf-8")
                            formatted_result = f"Give context about where the object is. Then inform the user that the {function_call.args["object_description"]} is located at {angle} degrees, {depth} meters away. Image Context: {decoded_str}"
                            # encoded_str = result[0]

                            # formatted_result = {
                            #     "response": decoded_str,
                            #     "value1": float(result[1]),
                            #     "value2": float(result[2])
                            # }

                            # Send properly formatted data
                            await self.session.send(
                                input=formatted_result, end_of_turn=True
                            )

                if response.data:
                    self.audio_in_queue.put_nowait(response.data)
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
            async with CLIENT.aio.live.connect(
                model=MODEL, config=CONFIG
            ) as session, asyncio.TaskGroup() as tg:
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

    async def find_entity(self, object_description: str) -> str:
        """
        Dummy implementation for a function call from Gemini.
        Replace this with your own object lookup logic.
        """

        print("Taking a snapshot")

        try:
            response = requests.get(SNAPSHOT_URL, timeout=5)

            if response.status_code == 200:
                image_data = np.frombuffer(response.content, np.uint8)
                img = cv2.imdecode(image_data, cv2.IMREAD_COLOR)
                image_path = os.path.join("snapshot.jpg")
                cv2.imwrite(image_path, img)

            else:
                print(f"Failed to get snapshot: {response.status_code}")

        except requests.exceptions.RequestException as e:
            print(f"Error: {e}")

        print("Sending data to GPU for processing...")
        if image_data is not NotImplementedError:
            with open(image_path, "rb") as img_file:
                files = {"image": img_file}
                data = {"label": object_description}

                try:
                    gpu_response = requests.post(
                        GPU_PROCESS_URL, data=data, files=files, timeout=5
                    )
                    gpu_response.raise_for_status()
                    gpu_json_response = gpu_response.json()

                    # Extract values
                    metric_depth = gpu_json_response.get("depth", None)
                    object_angle = gpu_json_response.get("heading", None)

                    print("Processing complete. Response:", gpu_response.json())

                except requests.exceptions.RequestException as e:
                    print(f"GPU processing error: {e}")

        file_names = [
            "original.jpg",
            "segmentation.jpg",
            "depth.jpg",
            "bounding_box.jpg",
            "mesh.glb",
            "segmented_mesh.glb",
        ]
        download_directory = (
            "C:/Users/jayqw/Documents/swd/Treehacks/Edith/frontend/public/uploads"
        )
        for name in file_names:
            file_path = os.path.join(download_directory, name)
            response = requests.get("http://172.24.75.90:8405/" + name, stream=True)
            with open(file_path, "wb") as file:
                for chunk in response.iter_content(chunk_size=1024):
                    if chunk:
                        file.write(chunk)

        bounding_box_b64 = None
        file_path = os.path.join(download_directory, "bounding_box.jpg")
        try:
            with open(file_path, "rb") as img_file:
                bounding_box_b64 = base64.b64encode(img_file.read()).decode("utf-8")

        except FileNotFoundError:
            print(f"File not found: {file_path}")

        input("Press Enter to continue...")

        # Heading, depth of object, and bounding box image
        try:
            return (bounding_box_b64, str(float(metric_depth) * 0.8), object_angle)
        except:
            return "Please try again. I was unable to find the object you requested.", 0, 0


if __name__ == "__main__":
    audio_chat = AudioChat()
    try:
        asyncio.run(audio_chat.run())
    except KeyboardInterrupt:
        print("Exiting on user interrupt.")
