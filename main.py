import tensorflow as tf
import numpy as np
import os
from PIL import Image
from uvicorn import run
from fastapi import UploadFile, File, FastAPI
from json import dumps

def load(image_file):
    """this function loads a blurry image into a float32tensor for further processing"""
    # Read and decode an image file to a uint8 tensor
    image = tf.io.read_file(image_file)
    image = tf.io.decode_jpeg(image)
    # ensure size is 256x256
    image = tf.image.resize(image, (256, 256))
    # recast to float32
    image = tf.cast(image, tf.float32)
    return image

app = FastAPI()
tf_saved_model = tf.keras.models.load_model('./litemodel')

@app.get("/")
async def root():
    return "Welcome to the Deblur API!"

@app.post("/local/deblur")
async def get_local_image_deblur(file: UploadFile = File(...)):
    """"this function receives an image that is locally hosted on the users computer
     and runs it through the deblur model"""

    ### acts as the load function, but directly works on the uploaded image
    img = file.file.read()
    img = tf.io.decode_jpeg(img)
    # ensure size is 256x256
    img = tf.image.resize(img, (256, 256))
    # recast to float32
    img = tf.cast(img, tf.float32)

    img = (img / 127.5) - 1
    img = np.array(np.expand_dims(img, 0))
    prediction = tf_saved_model(img, training=True)
    predicted_image = (prediction[0] + 1) * 127.5
    predicted_image = predicted_image.numpy()
    predicted_image = np.ndarray.tolist(predicted_image)
    predicted_image = dumps(predicted_image)
    return predicted_image

@app.post("/net/deblur")
async def get_net_image_deblur(image_link: str = ""):
    """"this function receives an image that is hosted on the world wide web
     and runs it through the deblur model"""
    if image_link == "":
        return {"message": "No image link provided"}
    image_path = tf.keras.utils.get_file(origin = image_link)
    img = load(image_path)

    img = (img / 127.5) - 1
    img = np.array(np.expand_dims(img, 0))
    prediction = tf_saved_model(img, training=True)
    predicted_image = (prediction[0] + 1) * 127.5
    predicted_image = predicted_image.numpy()
    predicted_image = np.ndarray.tolist(predicted_image)
    predicted_image = dumps(predicted_image)
    return predicted_image

if __name__ == "__main__":
	port = int(os.environ.get('PORT', 5000))
	run(app, host="0.0.0.0", port=port)