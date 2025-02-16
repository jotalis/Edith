import smbus2
import time
import math
import numpy as np
import RPi.GPIO as GPIO

# MPU6050 Registers
MPU6050_ADDR = 0x68  # Default I2C address
PWR_MGMT_1 = 0x6B
ACCEL_XOUT_H = 0x3B
GYRO_XOUT_H = 0x43

# Initialize I2C
bus = smbus2.SMBus(1)  # Use bus 1 for Raspberry Pi

# Wake up MPU6050
bus.write_byte_data(MPU6050_ADDR, PWR_MGMT_1, 0)

# GPIO setup
GPIO.setmode(GPIO.BCM)
motor_pin_1 = 12  # Pin for motor 1
motor_pin_2 = 13  # Pin for motor 2
GPIO.setup(motor_pin_1, GPIO.OUT)
GPIO.setup(motor_pin_2, GPIO.OUT)

def read_raw_data(addr):
    """ Read two bytes of data and convert to signed value """
    high = bus.read_byte_data(MPU6050_ADDR, addr)
    low = bus.read_byte_data(MPU6050_ADDR, addr + 1)
    value = (high << 8) | low
    if value > 32768:
        value -= 65536
    return value

def get_motion_data():
    """ Read accelerometer and gyroscope data """
    acc_x = read_raw_data(ACCEL_XOUT_H) / 16384.0  # Normalize to g
    acc_y = read_raw_data(ACCEL_XOUT_H + 2) / 16384.0
    acc_z = read_raw_data(ACCEL_XOUT_H + 4) / 16384.0

    gyro_x = read_raw_data(GYRO_XOUT_H) / 131.0  # Convert to degrees/sec
    gyro_y = read_raw_data(GYRO_XOUT_H + 2) / 131.0
    gyro_z = read_raw_data(GYRO_XOUT_H + 4) / 131.0

    return acc_x, acc_y, acc_z, gyro_x, gyro_y, gyro_z

def compute_orientation(acc_x, acc_y, acc_z):
    """ Compute pitch and roll from accelerometer data """
    pitch = math.degrees(math.atan2(acc_y, math.sqrt(acc_x**2 + acc_z**2)))
    roll = math.degrees(math.atan2(-acc_x, acc_z))
    return pitch, roll

def adjust_vibration_intensity(current_yaw, max_yaw, min_amplitude, max_amplitude, target_angle):
    """ Adjust vibration intensity based on yaw angle """
    difference = target_angle - current_yaw
    normalized_difference = max(0, min(max_yaw, abs(difference))) / max_yaw
    vibration_intensity = min_amplitude + (normalized_difference * (max_amplitude - min_amplitude))
    return vibration_intensity

# Complementary filter variables
dt = 0.25  # Time step
alpha = 0.98  # Complementary filter coefficient
pitch = roll = yaw = 0

# Vibration parameters
max_yaw = 180  # Maximum yaw angle
min_amplitude = 0  # Minimum vibration intensity
max_amplitude = 100  # Maximum vibration intensity
target_angle = 0  # Desired target angle

# Initialize PWM for motors
pwm_motor_1 = GPIO.PWM(motor_pin_1, 100)  # 100 Hz frequency
pwm_motor_2 = GPIO.PWM(motor_pin_2, 100)
pwm_motor_1.start(0)  # Start with 0% duty cycle
pwm_motor_2.start(0)  # Start with 0% duty cycle

print("Reading MPU6050 Data...")
try:
    while True:
        acc_x, acc_y, acc_z, gyro_x, gyro_y, gyro_z = get_motion_data()
        
        # Compute accelerometer-based angles
        acc_pitch, acc_roll = compute_orientation(acc_x, acc_y, acc_z)

        # Integrate gyroscope data for angle estimation
        pitch = alpha * (pitch + gyro_x * dt) + (1 - alpha) * acc_pitch
        roll = alpha * (roll + gyro_y * dt) + (1 - alpha) * acc_roll
        yaw += gyro_z * dt  # No accelerometer correction for yaw

        # Adjust vibration intensity based on yaw
        vibration_intensity = adjust_vibration_intensity(yaw, max_yaw, min_amplitude, max_amplitude, target_angle)

        # Update the duty cycle for the motors
        pwm_motor_1.ChangeDutyCycle(vibration_intensity)
        pwm_motor_2.ChangeDutyCycle(vibration_intensity)

        print(f"Pitch: {pitch:.2f}, Roll: {roll:.2f}, Yaw: {yaw:.2f}, Vibration Intensity: {vibration_intensity:.2f}")
        time.sleep(dt)

except KeyboardInterrupt:
    pass
finally:
    pwm_motor_1.stop()
    pwm_motor_2.stop()
    GPIO.cleanup()