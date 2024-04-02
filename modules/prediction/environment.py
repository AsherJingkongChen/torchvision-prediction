def request_best_device():
    """
    A helper function to request the best device for the current environment
    """
    from torch import backends, cuda, device

    if cuda.is_available():
        return device("cuda")
    elif backends.mps.is_available():
        return device("mps")
    elif backends.mkldnn.is_available():
        return device("mkldnn")
    else:
        return device("cpu")
