import tensorflow as tf
import tensorflow_datasets as tfds
import matplotlib.pyplot as plt
import numpy as np

train, test = tfds.load('rock_paper_scissors',split=['train','test'],as_supervised=True)

def precrocess(images,labels):
    # rescaled_image= images.reshape()
    return images,labels

rescale = []
label = []
for images, labels in train.map(precrocess):
    rescale.append(images.numpy().astype("uint8")/255.0)
    label.append(labels)