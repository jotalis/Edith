import asyncio
import cv2
import numpy as np
import websockets
import time
import json
import io
from picamera2 import Picamera2
import cv2

# GPU WebSocket Server Address (Replace with actual GPU IP if remote)
SERVER = "ws://10.32.72.225:8765"  # Change to your GPU IP

async def send_video_stream():
    """Captures video frames using PiCamera and sends them to GPU over WebSocket."""
    try:
        async with websockets.connect(
            SERVER,
            max_size=200 * 1024 * 1024,  # Increase to 200MB limit
            max_queue=None
        ) as websocket:
            # Initialize PiCamera2
            picam2 = Picamera2()
            
            # # Configure camera for 854x480 resolution
            # config = picam2.create_preview_configuration(
            #     main={"size": (1440, 1080), "format": "RGB888"}
            # )
            # picam2.configure(config)
            
            # Start the camera
            picam2.start()
            print("Starting camera...")
            
            # Allow camera to warm up
            # print("Warming up camera...")
            # await asyncio.sleep(2)
            
            while True:
                # Capture frame
                frame = picam2.capture_array()
                
                # Display the original frame
                # cv2.imshow("Webcam Feed", frame)
                
                # Compress frame using JPEG encoding
                encode_params = [int(cv2.IMWRITE_JPEG_QUALITY), 60]
                _, buffer = cv2.imencode('.jpg', frame, encode_params)
                frame_bytes = buffer.tobytes()
                
                # Record start time before sending
                start_time = time.time()
                
                # Send compressed frame bytes to GPU over WebSocket
                await websocket.send(frame_bytes)

                # Remove the receiving and processing part
                # Calculate and display latency
                latency = (time.time() - start_time) * 1000  # Convert to milliseconds
                print(f"Latency: {latency:.2f}ms")

                # Break loop on 'q' key press
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

            picam2.stop()
            cv2.destroyAllWindows()
            
    except websockets.exceptions.ConnectionClosedError as e:
        print(f"Connection closed with error: {e}")
    except Exception as e:
        print(f"An error occurred: {e}")

# Debugging: Check the event loop state
print("Starting event loop")
asyncio.run(send_video_stream())
print("Event loop finished")
