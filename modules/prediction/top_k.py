from typing import Callable, Iterator, TypeVar

_T = TypeVar("_T")


class TopK:
    """
    Maintain top-k elements
    """

    def __init__(self, k: int):
        super().__init__()
        self.k = k
        self._list = []

    def update(self, element: _T, key: Callable[[_T], _T] | None = None) -> None:
        """
        1. Add a new element
        2. Sort elements
        3. Remove the last element if necessary
        """
        self._list.append(element)
        self._list.sort(key=key)
        if len(self._list) > self.k:
            self._list.pop()

    def __iter__(self) -> Iterator[_T]:
        return self._list.__iter__()
