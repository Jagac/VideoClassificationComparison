from keras.layers import *
from keras.models import Model
from keras.applications.vgg16 import VGG16
from keras.callbacks import EarlyStopping
from preprocess import model_evaluation_plot, to_categorical, build_dataset
from parameters import CLASSES_LIST, SEQUENCE_LENGTH, IMAGE_HEIGHT, IMAGE_WIDTH, DIR
from sklearn.model_selection import train_test_split
import time
from keras.optimizers import Nadam


features, labels, video_files_paths = build_dataset()
one_hot_encoded_labels = to_categorical(labels)
features_train, features_test, labels_train, labels_test = train_test_split(features, one_hot_encoded_labels, test_size= 0.25, shuffle=True)


input = Input(shape=(SEQUENCE_LENGTH, IMAGE_HEIGHT, IMAGE_WIDTH, 3))
vgg = VGG16(input_shape=(IMAGE_WIDTH, IMAGE_HEIGHT, 3), weights="imagenet",include_top=False)
cnn_1 = GlobalAveragePooling2D()(vgg.output)
cnn = Model(inputs=vgg.input, outputs=cnn_1)
cnn.trainable = False

frames = TimeDistributed(cnn)(input)
sequence = LSTM(256)(frames)
hidden_layer = Dense(1024, activation="relu")(sequence)
outputs = Dense(len(CLASSES_LIST), activation="softmax")(hidden_layer)
model = Model([input], outputs)
model.summary()

optimizer = Nadam(learning_rate = 0.002,beta_1 = 0.9, beta_2 = 0.999, epsilon = 1e-08, schedule_decay = 0.004)
model.compile(loss="categorical_crossentropy", optimizer=optimizer,metrics=["categorical_accuracy"]) 
early_stopping = EarlyStopping(monitor= 'val_loss', patience = 10, mode = 'min', restore_best_weights = True)

print("VGG created")
st = time.time()
model_history = model.fit(x = features_train, y = labels_train, epochs = 50, batch_size = 4, shuffle = True, validation_split = 0.2, callbacks = [early_stopping])
et = time.time()
print(f"Execution time: {et-st}")

model_evaluation_plot("Vgg16 Results", model_history) 
model_1_evaluate = model.evaluate(features_test, labels_test)