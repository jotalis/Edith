import time
import board
import busio
import math
import RPi.GPIO as GPIO
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

# GPIO setup for PWM
MOTOR_PIN = 13  # GPIO pin connected to the transistor base
GPIO.setmode(GPIO.BCM)
GPIO.setup(MOTOR_PIN, GPIO.OUT)
pwm = GPIO.PWM(MOTOR_PIN, 1000)  # Initialize PWM on MOTOR_PIN at 1000Hz
pwm.start(0)  # Start PWM with 0% duty cycle (motor off)

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

def map_value(value, from_min, from_max, to_min, to_max):
    # Map a value from one range to another with a cubic transformation
    from_range = from_max - from_min
    to_range = to_max - to_min
    scaled_value = float(value - from_min) / float(from_range)

    return to_min + (scaled_value * to_range)

def initialize_imu():
    """Initialize and return the IMU sensor."""
    i2c = busio.I2C(board.SCL, board.SDA)
    bno = BNO08X_I2C(i2c, address=0x4B)
    bno.enable_feature(BNO_REPORT_ACCELEROMETER)
    bno.enable_feature(BNO_REPORT_GYROSCOPE)
    bno.enable_feature(BNO_REPORT_MAGNETOMETER)
    bno.enable_feature(BNO_REPORT_ROTATION_VECTOR)
    return bno

try:
    bno = initialize_imu()  # Initialize the IMU once before the loop
    while True:
        # Read rotation vector data (quaternion)
        try: 
            quat_w, quat_x, quat_y, quat_z = bno.quaternion  # Get quaternion values
            print(f"Quaternion: w={quat_w}, x={quat_x}, y={quat_y}, z={quat_z}")  # Debug print

            yaw, pitch, roll = quaternion_to_euler(quat_w, quat_x, quat_y, quat_z)
            print(f"Yaw: {yaw:.2f}°, Pitch: {pitch:.2f}°, Roll: {roll:.2f}°")

            # Define the angle threshold
            angle_threshold = 5  # Example threshold in degrees

            # Check if the yaw angle is within the threshold
            if abs(yaw) > angle_threshold:
                # Map pitch angle to PWM duty cycle
                duty_cycle = map_value(yaw, -180, 180, 0, 100)
            else:
                # Stop the motor if the angle exceeds the threshold
                duty_cycle = 0

            pwm.ChangeDutyCycle(duty_cycle)
            print(f"Motor Duty Cycle: {duty_cycle:.2f}%")
        except Exception as e:
            print(f"Error: {e}. Reinitializing IMU...")  # Print the error message for debugging
            bno = initialize_imu()  # Reinitialize the IMU sensor

        time.sleep(0.1)

except KeyboardInterrupt:
    pass

finally:
    pwm.stop()
    GPIO.cleanup()
