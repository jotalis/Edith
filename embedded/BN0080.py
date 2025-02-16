import time
import board
import busio
from adafruit_bno08x.i2c import BNO08X_I2C
from adafruit_bno08x import (
    BNO_REPORT_ACCELEROMETER,
    BNO_REPORT_GYROSCOPE,
    BNO_REPORT_MAGNETOMETER,
)
import RPi.GPIO as GPIO

i2c = busio.I2C(board.SCL, board.SDA)

bno = BNO08X_I2C(i2c, address=0x4b)

bno.enable_feature(BNO_REPORT_ACCELEROMETER)
bno.enable_feature(BNO_REPORT_GYROSCOPE)
bno.enable_feature(BNO_REPORT_MAGNETOMETER)

# GPIO setup for vibration motors
MOTOR_PIN = 13  
GPIO.setmode(GPIO.BCM)
GPIO.setup(MOTOR_PIN, GPIO.OUT)


def calculate_rotation_angle(gyro_x, gyro_y, gyro_z, dt):
    angle_x = gyro_x * dt
    angle_y = gyro_y * dt
    angle_z = gyro_z * dt
    return angle_x, angle_y, angle_z

# Function to control motor vibration
def control_motor_vibration(angle_z, target_angle_z):
    # Calculate the difference from the target angle
    angle_diff = abs(target_angle_z - angle_z)
    # Calculate vibration amplitude based on angle difference
    amplitude = max(0, 100 - angle_diff)  # Example calculation
    # Set motor vibration amplitude
    pwm = GPIO.PWM(MOTOR_PIN, 100)  # Fixed frequency
    pwm.start(amplitude)  # Amplitude as duty cycle
    time.sleep(0.1)  # Vibration duration
    pwm.stop()

def initialize_imu():
    """Initialize and return the IMU sensor."""
    i2c = busio.I2C(board.SCL, board.SDA)
    bno = BNO08X_I2C(i2c, address=0x4B)
    bno.enable_feature(BNO_REPORT_ACCELEROMETER)
    bno.enable_feature(BNO_REPORT_GYROSCOPE)
    bno.enable_feature(BNO_REPORT_MAGNETOMETER)
    return bno

try:
    # Main loop to read sensor data and control motors
    target_angle_z = 90  # Example target angle for z-axis
    while True:
        try: 
            bno = initialize_imu()
        # Read acceleration data (m/s^2)
            accel_x, accel_y, accel_z = bno.acceleration
            print(f"Acceleration - X: {accel_x:.2f}, Y: {accel_y:.2f}, Z: {accel_z:.2f} m/s^2")
            # Read gyroscope data (radians/s)
            gyro_x, gyro_y, gyro_z = bno.gyro
            print(f"Gyroscope - X: {gyro_x:.2f}, Y: {gyro_y:.2f}, Z: {gyro_z:.2f} rad/s")
            # Read magnetometer data (microteslas)
            mag_x, mag_y, mag_z = bno.magnetic
            print(f"Magnetometer - X: {mag_x:.2f}, Y: {mag_y:.2f}, Z: {mag_z:.2f} uT")
            print()
            # Calculate rotation angle
            dt = 0.1  # Time interval in seconds
            angle_x, angle_y, angle_z = calculate_rotation_angle(gyro_x, gyro_y, gyro_z, dt)
            # Control motor vibration based on z-axis rotation angle
            control_motor_vibration(angle_z, target_angle_z)
            time.sleep(dt)
        except Exception as e:
            time.sleep(1)
except KeyboardInterrupt:
    pass
finally:
    GPIO.cleanup()