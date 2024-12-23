import numpy as np
import pandas as pd
import seaborn as sb
import matplotlib.pyplot as plt

from glob import glob
from PIL import Image
from keras.src.applications.resnet import ResNet50
from sklearn.model_selection import train_test_split

import tensorflow as tf
import keras as keras
from keras import layers
from functools import partial

AUTO = tf.data.experimental.AUTOTUNE
import warnings
warnings.filterwarnings('ignore')

images = glob('train_cancer/*/*.jpg')
print(len(images))

#replace backslash with forward slash to avoid unexpected errors
images = [path.replace('\\', '/') for path in images]
df = pd.DataFrame({'filepath': images})
df['label'] = df['filepath'].str.split('/', expand=True)[1]
print(df.head())

df['label_bin'] = np.where(df['label'].values == 'malignant', 1, 0)
print(df.head())

x = df['label'].value_counts()
plt.pie(x.values,
        labels=x.index,
        autopct='%1.1f%%')
plt.show()

for cat in df['label'].unique():
    temp = df[df['label'] == cat]

    index_list = temp.index
    fig, ax = plt.subplots(1, 4, figsize=(15, 5))
    fig.suptitle(f'Images for {cat} category . . . .', fontsize=20)
    for i in range(4):
        index = np.random.randint(0, len(index_list))
        index = index_list[index]
        data = df.iloc[index]

        image_path = data[0]

        img = np.array(Image.open(image_path))
        ax[i].imshow(img)
plt.tight_layout()
plt.show()

features = df['filepath']
target = df['label_bin']

X_train, X_val,\
    Y_train, Y_val = train_test_split(features, target,
                                      test_size=0.15,
                                      random_state=10)

print(X_train.shape, X_val.shape)

def decode_image(filepath, label=None):
    img = tf.io.read_file(filepath)
    img = tf.image.decode_jpeg(img)
    img = tf.image.resize(img, [224, 224])
    img = tf.cast(img, tf.float32) / 255.0

    # Compare label with integer 0 or 1 instead of string
    if label == 0:
        Label = 0
    else:
        Label = 1

    return img, Label

train_ds = (
    tf.data.Dataset
    .from_tensor_slices((X_train, Y_train))
    .map(decode_image, num_parallel_calls=AUTO)
    .batch(32)
    .prefetch(AUTO)
)

val_ds = (
    tf.data.Dataset
    .from_tensor_slices((X_val, Y_val))
    .map(decode_image, num_parallel_calls=AUTO)
    .batch(32)
    .prefetch(AUTO)
)


pre_trained_model = keras.applications.ResNet50(input_shape=(224, 224, 3),
    weights=None,include_top=False)

pre_trained_model.load_weights('resnet50_weights_tf_dim_ordering_tf_kernels_notop.h5')

for layer in pre_trained_model.layers:
    layer.trainable = False

# Define the new model
inputs = layers.Input(shape=(224, 224, 3))
x = pre_trained_model(inputs, training=False)  # Use the pre-trained model
x = layers.Flatten()(x)
x = layers.Dense(256, activation='relu')(x)
x = layers.BatchNormalization()(x)
x = layers.Dense(256, activation='relu')(x)
x = layers.Dropout(0.3)(x)
x = layers.BatchNormalization()(x)
outputs = layers.Dense(1, activation='sigmoid')(x)

model = keras.Model(inputs, outputs)

model.compile(
    loss=keras.losses.BinaryCrossentropy(from_logits=True),
    optimizer='adam',
    metrics=['AUC']
)

history = model.fit(train_ds,
                    validation_data=val_ds,
                    epochs=5,
                    verbose=1)
hist_df = pd.DataFrame(history.history)
print(hist_df.head())