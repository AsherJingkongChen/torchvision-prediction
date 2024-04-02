def request_best_device():
    """
    Request the best device for the current environment
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


def request_snapshot_path(number: int | None = None) -> str:
    """
    Requests the snapshot path with the given number

    ## Parameters
    - number (`int | None`)
        - Defaults to `None`, and creates a new snapshot.

    ## Returns
    - The snapshot path (`str`)
    """
    from pathlib import Path

    if not number:
        number = (
            sorted(
                map(
                    lambda p: int(p.name),
                    Path("snapshots").glob("index/*/"),
                )
            )[-1:]
            or [0]
        )[0] + 1

    path = Path(f"snapshots/index/{number}/")
    path.mkdir(parents=True, exist_ok=bool(number))
    return str(path)
