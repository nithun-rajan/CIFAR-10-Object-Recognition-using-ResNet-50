import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
from PIL import Image
import matplotlib.image as mpimg
import py7zr
from sklearn.preprocessing import LabelEncoder
import tensorflow as tf
from tensorflow import keras
from tensorflow import keras
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense, Dropout, Flatten, BatchNormalization, UpSampling2D
from tensorflow.keras.applications.resnet50 import ResNet50
from tensorflow.keras import optimizers



archive_path = '/Users/nithunsundarrajan/Downloads/cifar-10/train.7z'
extract_path = '/Users/nithunsundarrajan/Downloads/cifar-10/train/train'

# Extract if not already extracted
if not os.path.exists(extract_path):
    with py7zr.SevenZipFile(archive_path, mode='r') as z:
        z.extractall(path=extract_path)

filenames = os.listdir(extract_path)

#print(type(filenames))
#print(len(filenames))
#print(filenames[0:5])  

labels = pd.read_csv('trainLabels.csv')
#print(labels.shape)
#print(labels.head())

labels['label'].value_counts

le = LabelEncoder()
labels['label'] = le.fit_transform(labels['label'])

#print(labels['label'].value_counts())

# Print mapping of encoded values to original labels
for idx, class_name in enumerate(le.classes_):
    print(f"{idx}: {class_name}")


#print(labels[0:5])

id_list = list(labels['id'])

#converting images to numpy arrays
train_data = '/Users/nithunsundarrajan/Downloads/cifar-10/train/train'

images = []

for id in id_list:
    image = Image.open(os.path.join(train_data, f"{id}.png"))
    image = np.array(image)
    images.append(image)

print(type(images))
print(len(images))

X = np.array(images)
y = labels['label'].values

# Splitting the dataset into training and validation sets

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
# Print shapes of the datasets
print(f"X_train shape: {X_train.shape}")
print(f"X_test shape: {X_test.shape}")
print(f"y_train shape: {y_train.shape}")
print(f"y_test shape: {y_test.shape}")


#scaling the data
X_train = X_train / 255.0
X_test = X_test / 255.0

import tensorflow as tf
from tensorflow import keras

num_of_classes = len(le.classes_)   
print(f"Number of classes: {num_of_classes}")

#setting up the model
model = keras.Sequential([
    keras.layers.Flatten(input_shape=(32, 32, 3)),
    keras.layers.Dense(64, activation='relu'),
    keras.layers.Dense(10, activation='softmax'),
])
# Compiling the model
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

y = labels['label'].values
# Training the model
model.fit(X_train, y_train, epochs=10, validation_split = 0.1 )



convolutional_base = ResNet50(include_top=False, weights='imagenet', input_shape=(256, 256, 3))

model = keras.Sequential([
    keras.layers.UpSampling2D((2, 2)),
    keras.layers.UpSampling2D((2, 2)),
    keras.layers.UpSampling2D((2, 2)),
    ResNet50(include_top=False, weights='imagenet', input_shape=(256, 256, 3)),
    keras.layers.Flatten(),
    keras.layers.BatchNormalization(),
    keras.layers.Dense(128, activation='relu'),
    keras.layers.Dropout(0.5),
    keras.layers.BatchNormalization(),
    keras.layers.Dense(64, activation='relu'),
    keras.layers.Dropout(0.5),
    keras.layers.BatchNormalization(),
    keras.layers.Dense(num_of_classes, activation='softmax')
])

# Compiling the model
model.compile(optimizer=optimizers.RMSprop(learning_rate=2e-5),
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# Training the model
history = model.fit(X_train, y_train, epochs=10, validation_split=0.1)





