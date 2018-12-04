import os
import re
from shutil import copyfile
import numpy as np
path_unity = "../data/UnityEyes/"
path_train = "../data/UnityEyesTrain/"
path_val = "../data/UnityEyesVal/"
path_test = "../data/UnityEyesTest/"

test_size = 10000
validation_size = 10000

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


def create_new_dataset(ids, size, from_path, to_path):
    print("Generating {} samples...".format(size))
    samples = np.random.choice(list(set(ids)), size=size,
                                    replace=False)
    copy_samples(samples, from_path, to_path)
    return samples


n_duplicates = len(ids) - len(set(ids))
print("{} Duplicates found".format(n_duplicates))


test_ids = create_new_dataset(ids, test_size, path_unity, path_test)

unused_ids = [id for id in ids if id not in test_ids]

validation_ids = create_new_dataset(unused_ids, validation_size, path_unity, path_val)

unused_ids = [id for id in unused_ids if id not in validation_ids]
copy_samples(unused_ids, path_unity, path_train)

