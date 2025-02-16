import smbus2
import time
import math
import numpy as np

# MPU6050 Registers
MPU6050_ADDR = 0x68  # Default I2C address
PWR_MGMT_1 = 0x6B
ACCEL_XOUT_H = 0x3B
GYRO_XOUT_H = 0x43

# Initialize I2C
bus = smbus2.SMBus(1)  # Use bus 1 for Raspberry Pi

# Wake up MPU6050
bus.write_byte_data(MPU6050_ADDR, PWR_MGMT_1, 0)

def read_raw_data(addr):
    """ Read two bytes of data and convert to signed value """
    high = bus.read_byte_data(MPU6050_ADDR, addr)
    low = bus.read_byte_data(MPU6050_ADDR, addr+1)
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

# Complementary filter variables
dt = 0.1  # Time step
alpha = 0.98  # Complementary filter coefficient
pitch = roll = yaw = 0

print("Reading MPU6050 Data...")
while True:
    acc_x, acc_y, acc_z, gyro_x, gyro_y, gyro_z = get_motion_data()
    
    # Compute accelerometer-based angles
    acc_pitch, acc_roll = compute_orientation(acc_x, acc_y, acc_z)

    # Integrate gyroscope data for angle estimation
    pitch = alpha * (pitch + gyro_x * dt) + (1 - alpha) * acc_pitch
    roll = alpha * (roll + gyro_y * dt) + (1 - alpha) * acc_roll
    yaw += gyro_z * dt  # No accelerometer correction for yaw

    print(f"Pitch: {pitch:.2f}, Roll: {roll:.2f}, Yaw: {yaw:.2f}")
    time.sleep(dt)
