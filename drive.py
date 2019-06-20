import socketio
import eventlet
import base64
from PIL import Image
from model_generator import process_image, model, load
from io import BytesIO
import numpy as np


# create a Socket.IO server
sio = socketio.Server()
shape = (100, 100, 3)
model = model(load=True, shape=None, checkpoint="checkpoints/short_light.h1_4")

# event sent by the simulator
@sio.on('telemetry')
def telemetry(sid, data):
    if data:
        # The current steering angle of the car
        steering_angle = float(data["steering_angle"])
        # The current throttle of the car, how hard to push peddle
        throttle = float(data["throttle"])
        # The current speed of the car
        speed = float(data["speed"])
        # The current image from the center camera of the car
        image_bytes = BytesIO(base64.b64decode(data["image"]))
        image, _ = process_image(image_bytes, None, False)

        # Use your model to compute steering and throttle
        steer = 1
        
        throttle = 1 - (speed/22)
        sa = model.predict(np.array([image]))[0][0]
        # print(sa)
        # response to the simulator with a steer angle and throttle
        send(sa, throttle)
    else:
        # Edge case
        sio.emit('manual', data={}, skip_sid=True)


# event fired when simulator connect
@sio.on('connect')
def connect(sid, environ):
    print("connect ", sid)
    send(0, 0)


# to send steer angle and throttle to the simulator
def send(steer, throttle):
    sio.emit("steer", data={'steering_angle': str(steer), 'throttle': str(throttle)}, skip_sid=True)


# wrap with a WSGI application
app = socketio.WSGIApp(sio)

# simulator will connect to localhost:4567
if __name__ == '__main__':
    eventlet.wsgi.server(eventlet.listen(('', 4567)), app)
