import os
from shutil import copyfile


def listdir(path, prefix='', postfix='', return_prefix=True, return_postfix=True):
    """
    Lists all files in path that start with prefix and end with postfix.
    By default, this function returns all filenames. If you do not want to
    return the pre- or postfix, set the corresponding parameters to False.
    :param path:
    :param prefix:
    :param postfix:
    :param return_prefix:
    :param return_postfix:
    :return: list(str)
    """
    files = os.listdir(path)
    filtered_files = filter(lambda f: f.startswith(prefix) and f.endswith(postfix), files)
    idx_start = 0 if return_prefix else len(prefix) - 1
    idx_end = 0 if return_postfix else len(postfix) - 1
    return_files = set([f[idx_start:-idx_end-1] for f in filtered_files])
    return list(return_files)


def create_folder_if_not_exists(path):
    if not os.path.exists(path):
        os.mkdir(path)


def save_images(tensor_generated, path, image_names, suffix=''):
    assert len(tensor_generated) == len(image_names)
    for i in range(len(tensor_generated)):
        filepath = os.path.join(path, "{}{}.jpg".format(image_names[i], suffix))
        with open(filepath, 'wb') as f:
            f.write(tensor_generated[i])


def copy_json(image_ids, folder_input, folder_output):
    for image_id in image_ids:
        filename = "{}.json".format(image_id)
        path_from = os.path.join(folder_input, filename)
        path_to = os.path.join(folder_output, filename)
        copyfile(path_from, path_to)

