
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras.layers import *
from tensorflow.keras.models import Sequential

import matplotlib.pyplot as plt
import time

features, labels, video_files_paths = build_dataset()

one_hot_encoded_labels = to_categorical(labels)

features_train, features_test, labels_train, labels_test = train_test_split(features, one_hot_encoded_labels, test_size= 0.25, shuffle=True)

def model_evaluation_plot(model_history, metric_1, metric_2, plot_name): 
    # might have to modify this for second model to plot them agains each other
    x = model_history.history[metric_1]
    y = model_history.history[metric_2]
    epochs = range(len(x))
    plt.plot(epochs, x, 'blue', label = metric_1)
    plt.plot(epochs, y, 'red', label = metric_2)
    plt.title(f"{plot_name}")
    plt.legend()
    plt.savefig(f"{plot_name}")
    
    

def conv_lstm_model():
    # https://proceedings.neurips.cc/paper/2015/file/07563a3fe3bbe7e3ba84431ad9d055af-Paper.pdf
    # https://keras.io/api/layers/recurrent_layers/conv_lstm2d/
    model = Sequential()

    model.add(ConvLSTM2D(filters = 4, kernel_size = (3,3), activation = "tanh", data_format = "channels_last",
    recurrent_dropout = 0.2, return_sequences = True, input_shape = (SEQUENCE_LENGTH, IMAGE_HEIGHT, IMAGE_WIDTH, 3)))
    model.add(MaxPooling3D(pool_size = (1,2,2), padding = 'same', data_format = "channels_last"))
    model.add(TimeDistributed(Dropout(0.2)))

    model.add(ConvLSTM2D(filters = 8, kernel_size = (3,3), activation = "tanh", data_format = "channels_last",
    recurrent_dropout = 0.2, return_sequences = True))
    model.add(MaxPooling3D(pool_size = (1,2,2), padding = 'same', data_format = "channels_last"))
    model.add(TimeDistributed(Dropout(0.2)))

    model.add(ConvLSTM2D(filters = 14, kernel_size = (3,3), activation = "tanh", data_format = "channels_last",
    recurrent_dropout = 0.2, return_sequences = True))
    model.add(MaxPooling3D(pool_size = (1,2,2), padding = 'same', data_format = "channels_last"))
    model.add(TimeDistributed(Dropout(0.2)))

    model.add(ConvLSTM2D(filters = 16, kernel_size = (3,3), activation = "tanh", data_format = "channels_last",
    recurrent_dropout = 0.2, return_sequences = True))
    model.add(MaxPooling3D(pool_size = (1,2,2), padding = 'same', data_format = "channels_last"))
    
    model.add(Flatten())
    model.add(Dense(len(CLASSES_LIST), activation = "softmax"))

    with open('model_1_summary.txt', 'w') as f:
        model.summary(print_fn=lambda x: f.write(x + '\n'))

    return model

def long_term_conv_model():
    # https://arxiv.org/abs/1411.4389
    model = Sequential()

    model.add(TimeDistributed(Conv2D(4, (3,3), padding = 'same', activation = 'relu'), input_shape = (SEQUENCE_LENGTH, IMAGE_HEIGHT, IMAGE_WIDTH, 3)))
    model.add(TimeDistributed(MaxPooling2D((2, 2))))
    model.add(TimeDistributed(Dropout(0.2)))

    model.add(TimeDistributed(Conv2D(8, (3,3), padding = 'same', activation = 'relu')))
    model.add(TimeDistributed(MaxPooling2D((2,2))))
    model.add(TimeDistributed(Dropout(0.2)))

    model.add(TimeDistributed(Conv2D(14, (3,3), padding = 'same', activation = 'relu')))
    model.add(TimeDistributed(MaxPooling2D((2,2))))
    model.add(TimeDistributed(Dropout(0.2)))

    model.add(TimeDistributed(Conv2D(16, (3,3), padding = 'same', activation = 'relu')))
    model.add(TimeDistributed(MaxPooling2D((2,2))))
    
    model.add(TimeDistributed(Flatten()))
    model.add(LSTM(32))
    model.add(Dense(len(CLASSES_LIST), activation = "softmax"))

    with open('model_2_summary.txt', 'w') as f:
        model.summary(print_fn=lambda x: f.write(x + '\n'))

    return model

model_1 = conv_lstm_model()
model_1.compile(loss = 'categorical_crossentropy', optimizer = 'Adam', metrics = ['accuracy'])
print("Model 1 created")


model_1_history = model_1.fit(x = features_train, y = labels_train, epochs = 50, batch_size = 4, shuffle = True, validation_split = 0.2)


model_1_evaluate = model_1.evaluate(features_test, labels_test)
model_1.save("model1.h5")
model_evaluation_plot(model_1_history, "loss", "val_loss", "ConvLSTM Loss per Epoch") # to find if we are overfitting 
# Might have to add early stopping
#https://keras.io/api/callbacks/early_stopping/
model_evaluation_plot(model_1_history, "accuracy", "val_accuracy", "ConvLSTM Accuracy per Epoch")

model_2 = long_term_conv_model()
model_2.compile(loss = 'categorical_crossentropy', optimizer = 'Adam', metrics = ['accuracy'])
print("Model 2 created")


model_2_history = model_2.fit(x = features_train, y = labels_train, epochs = 50, batch_size = 4, shuffle = True, validation_split = 0.2)


model_2_evaluate = model_2.evaluate(features_test, labels_test)
model_2.save("model2.h5")
model_evaluation_plot(model_2_history, "loss", "val_loss", "LRCNN Loss per Epoch") 
model_evaluation_plot(model_2_history, "accuracy", "val_accuracy", "LRCNN Accuracy per Epoch")

