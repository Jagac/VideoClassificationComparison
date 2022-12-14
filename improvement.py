# take best performing model from modeltests.py and improve it
from modeltests import long_term_conv_model
from preprocess import model_evaluation_plot, to_categorical, build_dataset
from parameters import CLASSES_LIST, SEQUENCE_LENGTH, IMAGE_HEIGHT, IMAGE_WIDTH, DIR
from sklearn.model_selection import train_test_split
from tensorflow.keras.layers import TimeDistributed, Conv2D, MaxPooling2D, Dropout, Bidirectional, LSTM, Flatten, Dense, Layer
from tensorflow.keras.models import Sequential
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras import backend as K
import time

features, labels, video_files_paths = build_dataset()
one_hot_encoded_labels = to_categorical(labels)
features_train, features_test, labels_train, labels_test = train_test_split(features, one_hot_encoded_labels, test_size= 0.25, shuffle=True)

class AttentionLSTM(Layer):
    def __init__(self):
        super(AttentionLSTM, self).__init__()
        
    def build(self, input_shape):
        self.W = self.add_weight(name="att_weight", shape=(input_shape[-1], 1), initializer = "normal")
        self.b = self.add_weight(name="att_bias", shape=(input_shape[1], 1), initializer = "zeros")
        super(AttentionLSTM, self).build(input_shape)
        
    def call(self, x):
        function = K.tanh(K.dot(x, self.W) + self.b)
        activation = K.softmax(function, axis = 1)
        output = x * activation
        
        return K.sum(output, axis=1)

def long_term_conv_model_improved():
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
    model.add(Bidirectional(LSTM(32, return_sequences=True)))
    model.add(AttentionLSTM())
    model.add(Dense(len(CLASSES_LIST), activation = "softmax"))

    with open('LRCNN IMP.txt', 'w') as f:
        model.summary(print_fn=lambda x: f.write(x + '\n'))

    return model


model_1 = long_term_conv_model_improved()
early_stopping = EarlyStopping(monitor= 'val_loss', patience = 10, mode = 'min', restore_best_weights = True)
model_1.compile(loss = 'categorical_crossentropy', optimizer = 'Adam', metrics = ['accuracy'])

print("Improved model created")
st = time.time()
model_1_history = model_1.fit(x = features_train, y = labels_train, epochs = 50, batch_size = 4, shuffle = True, validation_split = 0.2, callbacks = [early_stopping])
et = time.time()
print(f"Execution time: {et-st}")

model_evaluation_plot("Attention + bi-LSTM LRCNN Results", model_1_history) 
model_1_evaluate = model_1.evaluate(features_test, labels_test)


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