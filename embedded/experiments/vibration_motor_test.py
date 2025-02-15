import RPi.GPIO as GPIO
import time

# Set GPIO mode
GPIO.setmode(GPIO.BCM)

# GPIO pin setup (23/13 are the two PINS)
motor_pin = 13  # Example GPIO pin

GPIO.setup(motor_pin, GPIO.OUT)
# Create PWM object with initial 100Hz frequency
pwm = GPIO.PWM(motor_pin, 100)
# Start PWM with 50% duty cycle (constant intensity)
pwm.start(50)

try:
    # High frequency vibration (500Hz) with 75% amplitude
    pwm.ChangeFrequency(250)
    pwm.ChangeDutyCycle(75)  # Adjust amplitude to 75%
    time.sleep(3)  # Run for 3 seconds
    
    # Stop vibration briefly
    pwm.ChangeDutyCycle(0)
    time.sleep(3)  # Pause for 3 seconds
    pwm.ChangeDutyCycle(50)  # Reset amplitude to 50%
    
    # Low frequency vibration (50Hz) with 25% amplitude
    pwm.ChangeFrequency(250)
    pwm.ChangeDutyCycle(10)  # Adjust amplitude to 25%
    time.sleep(3)  # Run for 3 seconds

except KeyboardInterrupt:
    # Clean up
    pwm.stop()
    GPIO.cleanup()