import utils
import os
from sklearn.model_selection import StratifiedShuffleSplit

path2data = "C:/Users/jagac/Downloads/Data"
sub_folder_jpg = "hmdb51_jpg"

path2ajpgs = os.path.join(path2data, sub_folder_jpg)

all_vids, all_labels, catgs = utils.get_vids(path2ajpgs) 
len(all_vids), len(all_labels), len(catgs)

all_vids[:1], all_labels[:3], catgs[:5]

labels_dict = {}
ind = 0
for uc in catgs:
    labels_dict[uc] = ind
    ind+=1

labels_dict 
num_classes =5
unique_ids = [id_ for id_, label in zip(all_vids,all_labels) if labels_dict[label]<num_classes]
unique_labels = [label for id_, label in zip(all_vids,all_labels) if labels_dict[label]<num_classes]
len(unique_ids),len(unique_labels)

sss = StratifiedShuffleSplit(n_splits=2, test_size=0.1, random_state=0)
train_indx, test_indx = next(sss.split(unique_ids, unique_labels))

train_ids = [unique_ids[ind] for ind in train_indx]
train_labels = [unique_labels[ind] for ind in train_indx]
print(len(train_ids), len(train_labels)) 

test_ids = [unique_ids[ind] for ind in test_indx]
test_labels = [unique_labels[ind] for ind in test_indx]
print(len(test_ids), len(test_labels))

train_ids[:5], train_labels[:5]
test_ids[:5], test_labels[:5]