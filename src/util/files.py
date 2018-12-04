import os


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


def save_images(tensor, path, image_names):
    assert tensor.size == len(image_names)
    for i in range(tensor.size):
        filepath = "{}.jpg".format(os.path.join(path, image_names[i]))
        with open(filepath, 'wb') as f:
            f.write(tensor[i])
