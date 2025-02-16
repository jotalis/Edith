import asyncio
import cv2
import numpy as np
import websockets
import time
import json
from picamera2 import Picamera2
import smbus
import math

# GPU WebSocket Server Address (Replace with actual GPU IP if remote)
SERVER = "ws://10.32.72.225:8765"  # Change to your GPU IP

# MPU6050 Registers and their addresses
PWR_MGMT_1 = 0x6B
SMPLRT_DIV = 0x19
CONFIG = 0x1A
GYRO_CONFIG = 0x1B
INT_ENABLE = 0x38
ACCEL_XOUT_H = 0x3B
GYRO_XOUT_H = 0x43

# Initialize the I2C bus
bus = smbus.SMBus(1)  # or bus = smbus.SMBus(0) for older Raspberry Pi models
Device_Address = 0x68  # MPU6050 device address

# Function to read raw data from the MPU6050
def read_raw_data(addr):
    high = bus.read_byte_data(Device_Address, addr)
    low = bus.read_byte_data(Device_Address, addr+1)
    value = ((high << 8) | low)
    if(value > 32768):
        value = value - 65536
    return value

# Function to calculate pitch from accelerometer data
def calculate_pitch(acc_x, acc_y, acc_z):
    pitch_acc = math.atan2(acc_y, math.sqrt(acc_x**2 + acc_z**2))
    return math.degrees(pitch_acc)

async def send_frame(websocket, frame_bytes):
    """Send frame bytes over WebSocket."""
    await websocket.send(frame_bytes)

async def send_pitch(websocket, pitch):
    """Send pitch data over WebSocket."""
    await websocket.send(str(pitch))

async def send_data_stream():
    """Captures video frames and IMU pitch data, then sends them to GPU over WebSocket."""
    try:
        async with websockets.connect(
            SERVER,
            max_size=200 * 1024 * 1024,  # Increase to 200MB limit
            max_queue=None
        ) as websocket:
            # Initialize PiCamera2
            picam2 = Picamera2()
            picam2.start()
            print("Starting camera...")

            # Initialize MPU6050
            bus.write_byte_data(Device_Address, SMPLRT_DIV, 7)
            bus.write_byte_data(Device_Address, PWR_MGMT_1, 1)
            bus.write_byte_data(Device_Address, CONFIG, 0)
            bus.write_byte_data(Device_Address, GYRO_CONFIG, 24)
            bus.write_byte_data(Device_Address, INT_ENABLE, 1)
            
            start_time = time.time()
            while True:
                # Capture frame
                frame = picam2.capture_array()

                # Read Accelerometer raw value
                acc_x = read_raw_data(ACCEL_XOUT_H)
                acc_y = read_raw_data(ACCEL_XOUT_H + 2)
                acc_z = read_raw_data(ACCEL_XOUT_H + 4)

                # Calculate pitch
                pitch = calculate_pitch(acc_x, acc_y, acc_z)

                # Compress frame using JPEG encoding
                encode_params = [int(cv2.IMWRITE_JPEG_QUALITY), 60]
                _, buffer = cv2.imencode('.jpg', frame, encode_params)
                frame_bytes = buffer.tobytes()

                # Record start time before sending
                start_time = time.time()

                # Use asyncio.gather to send frame and pitch concurrently
                await asyncio.gather(
                    send_frame(websocket, frame_bytes),
                    send_pitch(websocket, pitch)
                )

                # Calculate and display latency
                latency = (time.time() - start_time) * 1000  # Convert to milliseconds
                print(f"Latency: {latency:.2f}ms, Pitch: {pitch:.2f}")

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
asyncio.run(send_data_stream())
print("Event loop finished")