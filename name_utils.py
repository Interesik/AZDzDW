import os.path as path

def name_from_params(class_, params):
    params_string = "_".join([f"{k}={v}" for k,v in params.items()])
    return f"{class_.__name__}_{params_string}"

def create_file_path(classifier_wrapper, suffix, directory = ""):
    return path.abspath(path.join(directory, f"{classifier_wrapper.name}_{suffix}"))