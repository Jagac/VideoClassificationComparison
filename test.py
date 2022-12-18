from tensorflow.keras.layers import *
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.optimizers import Adam
import tensorflow as tf
import numpy as np
from tensorflow.keras.applications import VGG16
from tensorflow.keras.callbacks import EarlyStopping
from preprocess import model_evaluation_plot, to_categorical, build_dataset
from parameters import CLASSES_LIST, SEQUENCE_LENGTH, IMAGE_HEIGHT, IMAGE_WIDTH, DIR
from sklearn.model_selection import train_test_split
import time

features, labels, video_files_paths = build_dataset()
one_hot_encoded_labels = to_categorical(labels)
features_train, features_test, labels_train, labels_test = train_test_split(features, one_hot_encoded_labels, test_size= 0.25, shuffle=True)


IMG_SIZE=(IMAGE_HEIGHT, IMAGE_WIDTH, 3)
num_class = len(CLASSES_LIST)

def vgg_model():
    conv_base = VGG16(weights='imagenet', include_top=False, input_shape=(IMAGE_HEIGHT, IMAGE_WIDTH, 3))
    x = GlobalAveragePooling2D()(conv_base.output)
    base_model = Model(conv_base.input, x)
    return base_model

conv_base = vgg_model()
ip = Input(shape=(10,64,64,3))
t_conv = TimeDistributed(conv_base)(ip) 

t_lstm = LSTM(10, return_sequences=False)(t_conv)
f_softmax = Dense(num_class, activation='softmax')(t_lstm)

model = Model(ip, f_softmax)

with open('vgg.txt', 'w') as f:
    model.summary(print_fn=lambda x: f.write(x + '\n'))



early_stopping = EarlyStopping(monitor= 'val_loss', patience = 10, mode = 'min', restore_best_weights = True)
model.compile(loss = 'categorical_crossentropy', optimizer = 'Adam', metrics = ['accuracy'])

print("Model 1 created")
st = time.time()
model_history = model.fit(x = features_train, y = labels_train, epochs = 50, batch_size = 4, shuffle = True, validation_split = 0.2, callbacks = [early_stopping])
et = time.time()
print(f"Execution time: {et-st}")
model_1_evaluate = model.evaluate(features_test, labels_test)
model_evaluation_plot("ConvLSTM Results", model_history)