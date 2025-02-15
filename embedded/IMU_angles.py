import time
import board
import busio
import math
from adafruit_bno08x.i2c import BNO08X_I2C
from adafruit_bno08x import (
    BNO_REPORT_ACCELEROMETER,
    BNO_REPORT_GYROSCOPE,
    BNO_REPORT_MAGNETOMETER,
    BNO_REPORT_ROTATION_VECTOR,
)
# Initialize I2C bus
i2c = busio.I2C(board.SCL, board.SDA)
# Initialize BNO08X sensor
bno = BNO08X_I2C(i2c, address=0x4b)
# Enable desired features
bno.enable_feature(BNO_REPORT_ACCELEROMETER)
bno.enable_feature(BNO_REPORT_GYROSCOPE)
bno.enable_feature(BNO_REPORT_MAGNETOMETER)
bno.enable_feature(BNO_REPORT_ROTATION_VECTOR)

def quaternion_to_euler(q_w, q_x, q_y, q_z):
    # Roll (x-axis rotation)
    sinr_cosp = 2 * (q_w * q_x + q_y * q_z)
    cosr_cosp = 1 - 2 * (q_x * q_x + q_y * q_y)
    roll = math.atan2(sinr_cosp, cosr_cosp)

    # Pitch (y-axis rotation)
    sinp = 2 * (q_w * q_y - q_z * q_x)
    if abs(sinp) >= 1:
        pitch = math.copysign(math.pi / 2, sinp)  # Use 90 degrees if out of range
    else:
        pitch = math.asin(sinp)

    # Yaw (z-axis rotation)
    siny_cosp = 2 * (q_w * q_z + q_x * q_y)
    cosy_cosp = 1 - 2 * (q_y * q_y + q_z * q_z)
    yaw = math.atan2(siny_cosp, cosy_cosp)

    # Convert to degrees
    roll_deg = math.degrees(roll)
    pitch_deg = math.degrees(pitch)
    yaw_deg = math.degrees(yaw)

    return yaw_deg, pitch_deg, roll_deg

# Main loop to read sensor data
while True:
    try:
        # Read acceleration data (m/s^2)
        accel_x, accel_y, accel_z = bno.acceleration
        print(f"Acceleration - X: {accel_x:.2f}, Y: {accel_y:.2f}, Z: {accel_z:.2f} m/s^2")
        # Read gyroscope data (radians/s)
        gyro_x, gyro_y, gyro_z = bno.gyro
        print(f"Gyroscope - X: {gyro_x:.2f}, Y: {gyro_y:.2f}, Z: {gyro_z:.2f} rad/s")
        # Read magnetometer data (microteslas)
        mag_x, mag_y, mag_z = bno.magnetic
        print(f"Magnetometer - X: {mag_x:.2f}, Y: {mag_y:.2f}, Z: {mag_z:.2f} uT")
        # Read rotation vector data (yaw, pitch, roll)
        quat_w, quat_x, quat_y, quat_z = bno.quaternion  # Get quaternion values
        yaw, pitch, roll = quaternion_to_euler(quat_w, quat_x, quat_y, quat_z)
        
        print(f"Yaw: {yaw:.2f}°, Pitch: {pitch:.2f}°, Roll: {roll:.2f}°")
        
    except Exception as e:
        pass
    
    time.sleep(0.1)