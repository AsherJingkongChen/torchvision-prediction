from typing import Callable, Iterator, TypeVar

_T = TypeVar("_T")


class TopK:
    """
    Maintain top-k elements
    """

    def __init__(self, k: int):
        super().__init__()
        self.k = k
        self.list = []

    def update(self, element: _T, key: Callable[[_T], _T] | None = None) -> None:
        """
        1. Add a new element
        2. Sort elements
        3. Remove the last element if necessary
        """
        self.list.append(element)
        self.list.sort(key=key)
        if len(self.list) > self.k:
            self.list.pop()

    def __iter__(self) -> Iterator[_T]:
        return self.list.__iter__()
