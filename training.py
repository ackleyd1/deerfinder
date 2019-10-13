from __future__ import absolute_import, division, print_function, unicode_literals

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

from PIL import Image

import time

AUTOTUNE = tf.data.experimental.AUTOTUNE

CLASS_NAMES = np.array(['deer'])

IMG_HEIGHT = 224
IMG_WIDTH = 224
BATCH_SIZE = 64
TIME_STEPS = 1000

def get_label(file_path):
    # convert the path to a list of path components
    parts = tf.strings.split(file_path, "/")
    # The second to last is the class-directory
    return parts[-2] == CLASS_NAMES


def decode_img(img):
    # convert the compressed string to a 3D uint8 tensor
    img = tf.image.decode_jpeg(img, channels=3)
    # Use `convert_image_dtype` to convert to floats in the [0,1] range.
    img = tf.image.convert_image_dtype(img, tf.float32)
    # resize the image to the desired size.
    return tf.image.resize(img, [IMG_WIDTH, IMG_HEIGHT])


def process_path(file_path):
    label = get_label(file_path)
    # load the raw data from the file as a string
    img = tf.io.read_file(file_path)
    img = decode_img(img)
    return img, label


def prepare_for_training(ds, cache=True, shuffle_buffer_size=32):
  # This is a small dataset, only load it once, and keep it in memory.
  # use `.cache(filename)` to cache preprocessing work for datasets that don't
  # fit in memory.
  if cache:
    if isinstance(cache, str):
      ds = ds.cache(cache)
    else:
      ds = ds.cache()

  ds = ds.shuffle(buffer_size=shuffle_buffer_size)

  # Repeat forever
  ds = ds.repeat()

  ds = ds.batch(BATCH_SIZE)

  # `prefetch` lets the dataset fetch batches in the background while the model
  # is training.
  ds = ds.prefetch(buffer_size=AUTOTUNE)

  return ds


def timeit(ds, steps=TIME_STEPS):
  start = time.time()
  it = iter(ds)
  for i in range(steps):
        batch = next(it)
        if i%10 == 0:
            print('.',end='')
  print()
  end = time.time()

  duration = end-start
  print("{} batches: {} s".format(steps, duration))
  print("{:0.5f} Images/s".format(BATCH_SIZE*steps/duration))

# create datasets of image files
dataset = tf.data.Dataset.list_files("./training/*/?*.*")
#unclassified = tf.data.Dataset.list_files("./training/unclassified/?*.*")

# determine count of files
count = tf.data.experimental.cardinality(dataset)
steps_per_epoch = np.ceil(count/BATCH_SIZE)

# Set `num_parallel_calls` so multiple images are loaded/processed in parallel.
labeled_dataset = dataset.map(process_path, num_parallel_calls=AUTOTUNE)

# prepare the datasets for training and cache them
prepared_dataset = prepare_for_training(labeled_dataset, cache='./dataset.tfcache')

#timeit(prepared_dataset)

vgg_conv = tf.keras.applications.VGG16()
x = tf.keras.layers.Dense(1, activation='sigmoid', name='predictions')(vgg_conv.layers[-2].output)
model = tf.keras.Model(inputs=vgg_conv.input, outputs=x)
model.compile(loss=tf.keras.losses.binary_crossentropy, optimizer='adam', metrics=["accuracy"])

log = model.fit(prepared_dataset, epochs=2, verbose=2, steps_per_epoch=steps_per_epoch)
print(log)

# model = keras.Sequential([
#     keras.layers.Flatten(input_shape=(224, 224)),
#     keras.layers.Dense(128, activation='relu'),
#     keras.layers.Dense(10, activation='softmax')
# ])
