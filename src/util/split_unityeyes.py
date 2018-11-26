import os
import re
from shutil import copyfile
import numpy as np
path_unity = "../data/UnityEyes/"
path_train = "../data/UnityEyesTrain/"
path_test = "../data/UnityEyesTest/"

test_size = 20000

files = os.listdir(path_unity)
ids = [re.sub(r"[^\d]", "", f) for f in files if f.endswith(".jpg")]

def create_filepath(folder, id, suffix):
    filename = "{}.{}".format(id, suffix)
    return os.path.join(folder, filename)


def create_folder_if_not_exists(path):
    if not os.path.exists(path):
        os.mkdir(path)


def copy_samples(ids, path_from, path_to):
    create_folder_if_not_exists(path_to)
    print("Copying {} images from {} to {}".format(len(ids), path_from, path_to))
    for id in ids:
        file_jpg = create_filepath(path_from, id, 'jpg')
        file_json = create_filepath(path_from, id, 'json')
        copyfile(file_jpg, create_filepath(path_to, id, 'jpg'))
        copyfile(file_json, create_filepath(path_to, id, 'json'))
    print("Copied {} images".format(len(ids)))


n_duplicates = len(ids) - len(set(ids))
print("{} Duplicates found".format(n_duplicates))

print("Generating test samples...")
test_samples = np.random.choice(list(set(ids)), size=test_size, replace=False)
set_test = set(test_samples)
copy_samples(test_samples, path_unity, path_test)

print("Generating train samples")
train_samples = [id for id in ids if id not in test_samples]
copy_samples(train_samples, path_unity, path_train)

