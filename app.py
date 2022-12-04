import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

#batch_size = 32
#img_height = 224
#img_width = 224
#
from tensorflow import keras
#model = keras.models.load_model('path/to/location')
from keras.models import load_model

st.set_option('deprecation.showfileUploaderEncoding', False)

global model
#model = tf.keras.models.load_model('model.h5')

@st.cache(allow_output_mutation=True)
def load_model():
	model = tf.keras.models.load_model('./my_model.h5')
	return model


def predict_class(image, model):

	image = tf.cast(image, tf.float32)
	image = tf.image.resize(image, [224, 224])

	image = np.expand_dims(image, axis = 0)

	prediction = model.predict(image)

	return prediction


model = load_model()
st.title('Breast Cancer Image Segmentation  CNN')

file = st.file_uploader("Upload an image of a Breast", type=["jpg", "png"])


if file is None:
	st.text('Waiting for upload....')

else:
	slot = st.empty()
	slot.text('Running inference....')

	test_image = Image.open(file)

	st.image(test_image, caption="Input Image", width = 400)

	pred = predict_class(np.asarray(test_image), model)

	class_names = ['benign', 'malignant', 'normal']

	result = class_names[np.argmax(pred)]

	output = 'The image is a ' + result

	slot.text('Done')

	st.success(output)

