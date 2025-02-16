import smbus
import time
import math

# MPU6050 Registers and their addresses
PWR_MGMT_1 = 0x6B
SMPLRT_DIV = 0x19
CONFIG = 0x1A
GYRO_CONFIG = 0x1B
INT_ENABLE = 0x38
ACCEL_XOUT_H = 0x3B
GYRO_XOUT_H = 0x43

def read_raw_data(addr):
    # Accelero and Gyro value are 16-bit
    high = bus.read_byte_data(Device_Address, addr)
    low = bus.read_byte_data(Device_Address, addr+1)
    # Concatenate higher and lower value
    value = ((high << 8) | low)
    # To get signed value from mpu6050
    if(value > 32768):
        value = value - 65536
    return value

def convert_to_mps2(raw_data, sensitivity=16384):
    """
    Convert raw accelerometer data to m/s².

    :param raw_data: The raw accelerometer data from the MPU6050.
    :param sensitivity: The sensitivity scale factor (default is 16384 for ±2g).
    :return: The acceleration in m/s².
    """
    g_force = raw_data / sensitivity
    acceleration_mps2 = g_force * 9.81
    return acceleration_mps2

def calculate_yaw_pitch_roll(acc_x, acc_y, acc_z, gyro_x, gyro_y, gyro_z, dt):
    """
    Calculate yaw, pitch, and roll from accelerometer and gyroscope data.

    :param acc_x: Accelerometer X-axis data.
    :param acc_y: Accelerometer Y-axis data.
    :param acc_z: Accelerometer Z-axis data.
    :param gyro_x: Gyroscope X-axis data.
    :param gyro_y: Gyroscope Y-axis data.
    :param gyro_z: Gyroscope Z-axis data.
    :param dt: Time interval between measurements.
    :return: Yaw, pitch, and roll angles.
    """
    # Convert gyroscope degrees/sec to radians/sec
    gyro_x_rad = math.radians(gyro_x)
    gyro_y_rad = math.radians(gyro_y)
    gyro_z_rad = math.radians(gyro_z)

    # Calculate pitch and roll from accelerometer data
    pitch_acc = math.atan2(acc_y, math.sqrt(acc_x**2 + acc_z**2))
    roll_acc = math.atan2(-acc_x, acc_z)

    # Integrate gyroscope data to get angles
    pitch_gyro = gyro_x_rad * dt
    roll_gyro = gyro_y_rad * dt
    yaw_gyro = gyro_z_rad * dt

    # Combine accelerometer and gyroscope data
    pitch = pitch_acc * 0.98 + pitch_gyro * 0.02
    roll = roll_acc * 0.98 + roll_gyro * 0.02
    yaw = yaw_gyro  # Yaw is primarily derived from gyroscope

    return math.degrees(yaw), math.degrees(pitch), math.degrees(roll)

# Initialize the I2C bus
bus = smbus.SMBus(1)  # or bus = smbus.SMBus(0) for older Raspberry Pi models
Device_Address = 0x68  # MPU6050 device address

# Write to sample rate register
bus.write_byte_data(Device_Address, SMPLRT_DIV, 7)

# Write to power management register
bus.write_byte_data(Device_Address, PWR_MGMT_1, 1)

# Write to Configuration register
bus.write_byte_data(Device_Address, CONFIG, 0)

# Write to Gyro configuration register
bus.write_byte_data(Device_Address, GYRO_CONFIG, 24)

# Write to interrupt enable register
bus.write_byte_data(Device_Address, INT_ENABLE, 1)

print("Reading Data from MPU6050")

previous_time = time.time()

while True:
    # Read Accelerometer raw value
    acc_x = read_raw_data(ACCEL_XOUT_H)
    acc_y = read_raw_data(ACCEL_XOUT_H + 2)
    acc_z = read_raw_data(ACCEL_XOUT_H + 4)

    # Read Gyroscope raw value
    gyro_x = read_raw_data(GYRO_XOUT_H)
    gyro_y = read_raw_data(GYRO_XOUT_H + 2)
    gyro_z = read_raw_data(GYRO_XOUT_H + 4)

    # Full scale range +/- 250 degree/C as per sensitivity scale factor
    Ax = acc_x / 16384.0
    Ay = acc_y / 16384.0
    Az = acc_z / 16384.0

    Gx = gyro_x / 131.0
    Gy = gyro_y / 131.0
    Gz = gyro_z / 131.0

    current_time = time.time()
    dt = current_time - previous_time
    previous_time = current_time

    # Calculate yaw, pitch, and roll
    yaw, pitch, roll = calculate_yaw_pitch_roll(acc_x, acc_y, acc_z, Gx, Gy, Gz, dt)

    print(f"Ax={Ax:.2f} Ay={Ay:.2f} Az={Az:.2f} Gx={Gx:.2f} Gy={Gy:.2f} Gz={Gz:.2f} Yaw={yaw:.2f} Pitch={pitch:.2f} Roll={roll:.2f}")
    
    time.sleep(1)
