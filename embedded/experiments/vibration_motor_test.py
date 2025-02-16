import RPi.GPIO as GPIO
import time

# Set GPIO mode
GPIO.setmode(GPIO.BCM)

# GPIO pin setup for the motor
motor_pin_1 = 12  # First motor GPIO pin

GPIO.setup(motor_pin_1, GPIO.OUT)

# Create PWM object with initial 100Hz frequency
pwm1 = GPIO.PWM(motor_pin_1, 100)

# Start PWM with 50% duty cycle (constant intensity)
pwm1.start(50)

try:
    # High frequency vibration (500Hz) with 75% amplitude
    pwm1.ChangeFrequency(250)
    pwm1.ChangeDutyCycle(75)  # Adjust amplitude to 75%
    time.sleep(3)  # Run for 3 seconds
    
    # Stop vibration briefly
    pwm1.ChangeDutyCycle(0)
    time.sleep(3)  # Pause for 3 seconds
    pwm1.ChangeDutyCycle(50)  # Reset amplitude to 50%
    
    # Low frequency vibration (50Hz) with 25% amplitude
    pwm1.ChangeFrequency(250)
    pwm1.ChangeDutyCycle(10)  # Adjust amplitude to 25%
    time.sleep(3)  # Run for 3 seconds

except KeyboardInterrupt:
    # Clean up
    pwm1.stop()
    GPIO.cleanup()