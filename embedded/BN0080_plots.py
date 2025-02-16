import time
import board
import busio
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import numpy as np
from adafruit_bno08x.i2c import BNO08X_I2C
from adafruit_bno08x import (
    BNO_REPORT_ACCELEROMETER,
    BNO_REPORT_GYROSCOPE,
    BNO_REPORT_MAGNETOMETER,
)

# Initialize I2C bus and sensor
i2c = busio.I2C(board.SCL, board.SDA)
bno = BNO08X_I2C(i2c, address=0x4b)

# Enable desired features
bno.enable_feature(BNO_REPORT_ACCELEROMETER)
bno.enable_feature(BNO_REPORT_GYROSCOPE)
bno.enable_feature(BNO_REPORT_MAGNETOMETER)

# Create figure and subplots
fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(10, 8))
plt.tight_layout(pad=3.0)

# Initialize data lists
max_points = 100
time_data = []
accel_x_data, accel_y_data, accel_z_data = [], [], []
gyro_x_data, gyro_y_data, gyro_z_data = [], [], []
mag_x_data, mag_y_data, mag_z_data = [], [], []

# Set up plots
ax1.set_title('Acceleration vs Time')
ax1.set_ylabel('Acceleration (m/s²)')
ax2.set_title('Gyroscope vs Time')
ax2.set_ylabel('Angular Velocity (rad/s)')
ax3.set_title('Magnetometer vs Time')
ax3.set_ylabel('Magnetic Field (µT)')
ax3.set_xlabel('Time (s)')

# Initialize lines
line_ax = ax1.plot([], [], 'r-', [], [], 'g-', [], [], 'b-')
line_gy = ax2.plot([], [], 'r-', [], [], 'g-', [], [], 'b-')
line_mag = ax3.plot([], [], 'r-', [], [], 'g-', [], [], 'b-')

# Add legends
ax1.legend(['X', 'Y', 'Z'])
ax2.legend(['X', 'Y', 'Z'])
ax3.legend(['X', 'Y', 'Z'])

start_time = time.time()

def update(frame):
    current_time = time.time() - start_time
    
    # Read sensor data
    accel_x, accel_y, accel_z = bno.acceleration
    gyro_x, gyro_y, gyro_z = bno.gyro
    mag_x, mag_y, mag_z = bno.magnetic

    print(f"Acceleration: X: {accel_x:.2f} m/s², Y: {accel_y:.2f} m/s², Z: {accel_z:.2f} m/s²")
    print(f"Gyroscope: X: {gyro_x:.2f} rad/s, Y: {gyro_y:.2f} rad/s, Z: {gyro_z:.2f} rad/s")
    print(f"Magnetometer: X: {mag_x:.2f} µT, Y: {mag_y:.2f} µT, Z: {mag_z:.2f} µT")
    
    # Update data lists
    time_data.append(current_time)
    accel_x_data.append(accel_x)
    accel_y_data.append(accel_y)
    accel_z_data.append(accel_z)
    gyro_x_data.append(gyro_x)
    gyro_y_data.append(gyro_y)
    gyro_z_data.append(gyro_z)
    mag_x_data.append(mag_x)
    mag_y_data.append(mag_y)
    mag_z_data.append(mag_z)
    
    # Limit data points
    if len(time_data) > max_points:
        time_data.pop(0)
        accel_x_data.pop(0)
        accel_y_data.pop(0)
        accel_z_data.pop(0)
        gyro_x_data.pop(0)
        gyro_y_data.pop(0)
        gyro_z_data.pop(0)
        mag_x_data.pop(0)
        mag_y_data.pop(0)
        mag_z_data.pop(0)
    
    # Update line data
    for i, line in enumerate(line_ax):
        line.set_data(time_data, [accel_x_data, accel_y_data, accel_z_data][i])
    for i, line in enumerate(line_gy):
        line.set_data(time_data, [gyro_x_data, gyro_y_data, gyro_z_data][i])
    for i, line in enumerate(line_mag):
        line.set_data(time_data, [mag_x_data, mag_y_data, mag_z_data][i])
    
    # Adjust axes limits
    for ax in [ax1, ax2, ax3]:
        ax.relim()
        ax.autoscale_view()
        ax.set_xlim(max(0, current_time - 10), current_time)

# Create animation
ani = FuncAnimation(fig, update, interval=50, blit=False)
plt.show()
