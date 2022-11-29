from preprocess import *
from sklearn.model_selection import train_test_split

features, labels, video_files_paths = build_dataset()

one_hot_encoded_labels = to_categorical(labels)

features_train, features_test, labels_train, labels_test = train_test_split(features, one_hot_encoded_labels,
                                                                            test_size= 0.25, shuffle=True)

