from preprocess import model_evaluation_plot, to_categorical, build_dataset
from parameters import CLASSES_LIST, SEQUENCE_LENGTH, IMAGE_HEIGHT, IMAGE_WIDTH, DIR
from sklearn.model_selection import train_test_split
from tensorflow.keras.layers import TimeDistributed, Conv2D, MaxPooling2D, Dropout, LSTM, Flatten, Dense, ConvLSTM2D, MaxPooling3D, Conv3D
from tensorflow.keras.models import Sequential
from tensorflow.keras.callbacks import EarlyStopping
import time

features, labels, video_files_paths = build_dataset()
one_hot_encoded_labels = to_categorical(labels)
features_train, features_test, labels_train, labels_test = train_test_split(features, one_hot_encoded_labels, test_size= 0.25, shuffle=True)

def conv_lstm_model():
    # https://keras.io/api/layers/recurrent_layers/conv_lstm2d/
    
    model = Sequential()

    model.add(ConvLSTM2D(filters = 8, kernel_size = (3,3), activation = "tanh", data_format = "channels_last",
    recurrent_dropout = 0.25, return_sequences = True, input_shape = (SEQUENCE_LENGTH, IMAGE_HEIGHT, IMAGE_WIDTH, 3)))
    model.add(MaxPooling3D(pool_size = (1,2,2), padding = 'same', data_format = "channels_last"))
    model.add(TimeDistributed(Dropout(0.5)))

    model.add(ConvLSTM2D(filters = 12, kernel_size = (3,3), activation = "tanh", data_format = "channels_last",
    recurrent_dropout = 0.25, return_sequences = True))
    model.add(MaxPooling3D(pool_size = (1,2,2), padding = 'same', data_format = "channels_last"))
    model.add(TimeDistributed(Dropout(0.5)))

    model.add(ConvLSTM2D(filters = 16, kernel_size = (3,3), activation = "tanh", data_format = "channels_last",
    recurrent_dropout = 0.25, return_sequences = True))
    model.add(MaxPooling3D(pool_size = (1,2,2), padding = 'same', data_format = "channels_last"))
    model.add(TimeDistributed(Dropout(0.5)))

    model.add(ConvLSTM2D(filters = 20, kernel_size = (3,3), activation = "tanh", data_format = "channels_last",
    recurrent_dropout = 0.25, return_sequences = True))
    model.add(MaxPooling3D(pool_size = (1,2,2), padding = 'same', data_format = "channels_last"))
    
    model.add(Flatten())
    model.add(Dense(len(CLASSES_LIST), activation = "softmax"))

    with open('ConvLSTM.txt', 'w') as f:
        model.summary(print_fn=lambda x: f.write(x + '\n'))

    return model

model_1 = conv_lstm_model()
early_stopping = EarlyStopping(monitor= 'val_loss', patience = 10, mode = 'min', restore_best_weights = True)
model_1.compile(loss = 'categorical_crossentropy', optimizer = 'Adam', metrics = ['accuracy'])

print("Model 1 created")
st = time.time()
model_1_history = model_1.fit(x = features_train, y = labels_train, epochs = 50, batch_size = 4, shuffle = True, validation_split = 0.2, callbacks = [early_stopping])
et = time.time()
print(f"Execution time: {et-st}")

model_1_evaluate = model_1.evaluate(features_test, labels_test)
model_evaluation_plot("ConvLSTM Results", model_1_history) 


def long_term_conv_model():
    # https://arxiv.org/abs/1411.4389
    model = Sequential()

    model.add(TimeDistributed(Conv2D(8, (3,3), padding = 'same', activation = 'relu'), input_shape = (SEQUENCE_LENGTH, IMAGE_HEIGHT, IMAGE_WIDTH, 3)))
    model.add(TimeDistributed(MaxPooling2D((2, 2))))
    model.add(TimeDistributed(Dropout(0.2)))

    model.add(TimeDistributed(Conv2D(12, (3,3), padding = 'same', activation = 'relu')))
    model.add(TimeDistributed(MaxPooling2D((2,2))))
    model.add(TimeDistributed(Dropout(0.2)))

    model.add(TimeDistributed(Conv2D(16, (3,3), padding = 'same', activation = 'relu')))
    model.add(TimeDistributed(MaxPooling2D((2,2))))
    model.add(TimeDistributed(Dropout(0.2)))

    model.add(TimeDistributed(Conv2D(20, (3,3), padding = 'same', activation = 'relu')))
    model.add(TimeDistributed(MaxPooling2D((2,2))))
    
    model.add(TimeDistributed(Flatten()))
    model.add(LSTM(32))
    model.add(Dense(len(CLASSES_LIST), activation = "softmax"))

    with open('LRCNN.txt', 'w') as f:
        model.summary(print_fn=lambda x: f.write(x + '\n'))

    return model

model_2 = long_term_conv_model()
early_stopping = EarlyStopping(monitor= 'val_loss', patience = 10, mode = 'min', restore_best_weights = True)
model_2.compile(loss = 'categorical_crossentropy', optimizer = 'Adam', metrics = ['accuracy'])

print("Model 2 created")
st = time.time()
model_2_history = model_2.fit(x = features_train, y = labels_train, epochs = 50, batch_size = 4, shuffle = True, validation_split = 0.2, callbacks = [early_stopping])
et = time.time()
print(f"Execution time: {et-st}")

model_2_evaluate = model_2.evaluate(features_test, labels_test)
model_evaluation_plot("LRCNN Results", model_2_history) 


def conv3d_model():
    model = Sequential()

    model.add(Conv3D(14, (3,3,3), activation='relu', input_shape = (SEQUENCE_LENGTH, IMAGE_HEIGHT, IMAGE_WIDTH, 3)))
    model.add(MaxPooling3D(pool_size = (1,2,2), padding = 'same'))
    model.add(TimeDistributed(Dropout(0.5)))

    model.add(Conv3D(16, (3,3,3), activation='relu'))
    model.add(MaxPooling3D(pool_size = (1,2,2), padding = 'same'))
    model.add(TimeDistributed(Dropout(0.5)))

    model.add(Conv3D(18, (3,3,3), activation='relu'))
    model.add(MaxPooling3D(pool_size = (1,2,2), padding = 'same'))
    model.add(TimeDistributed(Dropout(0.5)))

    model.add(Conv3D(22, (3,3,3), activation='relu'))
    model.add(MaxPooling3D(pool_size = (1,2,2), padding = 'same'))
    
    model.add((Flatten()))
    model.add(Dense(len(CLASSES_LIST), activation = "softmax"))

    with open('conv3d.txt', 'w') as f:
        model.summary(print_fn=lambda x: f.write(x + '\n'))

    return model

model_3 = conv3d_model()
early_stopping = EarlyStopping(monitor= 'val_loss', patience = 10, mode = 'min', restore_best_weights = True)
model_3.compile(loss = 'categorical_crossentropy', optimizer = 'Adam', metrics = ['accuracy'])

print("Model 3 created")
st = time.time()
model_3_history = model_3.fit(x = features_train, y = labels_train, epochs = 50, batch_size = 4, shuffle = True, validation_split = 0.2, callbacks = [early_stopping])
et = time.time()
print(f"Execution time: {et-st}")

model_3_evaluate = model_3.evaluate(features_test, labels_test)
model_evaluation_plot("Conv3D Results", model_3_history) 


