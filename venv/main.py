import tensorflow as tf
import tensorflow_datasets as tfds
import matplotlib.pyplot as plt
import numpy as np

train,test = tfds.load('fashion_mnist',split=['train','test'],as_supervised=True)

train_images = []
train_labels = []

for images,labels in train:
  train_images.append(images.numpy().astype("uint8")/255.0)
  train_labels.append(labels)



test_images = []
test_labels = []

for images,labels in test:
  test_images.append(images.numpy().astype("uint8")/255.0)
  test_labels.append(labels)

train_images = np.array(train_images)
train_labels = np.array(train_labels)
test_images = np.array(test_images)
test_labels = np.array(test_labels)

train_images = tf.image.random_flip_left_right(train_images)
train_images = tf.image.random_flip_up_down(train_images)


model = tf.keras.Sequential([tf.keras.layers.Input(shape=(28,28,1)),
                             tf.keras.layers.Conv2D(124,(3,3),activation='relu'),
                             tf.keras.layers.MaxPooling2D(2,2),
                             tf.keras.layers.Conv2D(124,(3,3),activation='relu'),
                             tf.keras.layers.MaxPooling2D(2,2),
                             tf.keras.layers.Flatten(),
                             tf.keras.layers.Dense(512,activation=tf.keras.activations.relu),
                             tf.keras.layers.Dropout(0.2),
                             tf.keras.layers.Dense(256,activation='relu'),
                             tf.keras.layers.Dropout(0.2),
                             tf.keras.layers.Dense(10,activation='softmax')])

model.compile(optimizer='adam',loss = 'sparse_categorical_crossentropy',metrics=['accuracy'])

history = model.fit(train_images,train_labels,epochs=80,steps_per_epoch=12)

model.save('fashion_mnist_model.h5')

model.evaluate(test_images,test_labels)
