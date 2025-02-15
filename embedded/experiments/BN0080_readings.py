import time
import board
import busio
from adafruit_bno08x.i2c import BNO08X_I2C
from adafruit_bno08x import (
    BNO_REPORT_ACCELEROMETER,
    BNO_REPORT_GYROSCOPE,
    BNO_REPORT_MAGNETOMETER,
)
# Initialize I2C bus
i2c = busio.I2C(board.SCL, board.SDA)
# Initialize BNO08X sensor
bno = BNO08X_I2C(i2c, address=0x4b)
# Enable desired features
bno.enable_feature(BNO_REPORT_ACCELEROMETER)
bno.enable_feature(BNO_REPORT_GYROSCOPE)
bno.enable_feature(BNO_REPORT_MAGNETOMETER)
# Main loop to read sensor data
while True:
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
    time.sleep(1)