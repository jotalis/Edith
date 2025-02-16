import cv2
from flask import Flask, Response
from picamera2 import Picamera2

app = Flask(__name__)

# Initialize Picamera2
picam2 = Picamera2()
picam2.start()
print("Starting camera...")


@app.route("/snapshot")
def snapshot():
    try:
        frame = picam2.capture_array()  # Capture image as NumPy array

        # Encode frame as JPEG
        encode_params = [int(cv2.IMWRITE_JPEG_QUALITY), 60]
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        success, buffer = cv2.imencode(".jpg", frame_rgb, encode_params)

        if not success:
            return Response("Failed to encode image", status=500)

        frame_bytes = buffer.tobytes()
        return Response(frame_bytes, mimetype="image/jpeg")

    except Exception as e:
        return Response(f"Error: {str(e)}", status=500)


@app.route("/angle")
def angle():
    angle_value = 45.0  # Example float value
    return Response(str(angle_value), mimetype="text/plain")


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=False)
