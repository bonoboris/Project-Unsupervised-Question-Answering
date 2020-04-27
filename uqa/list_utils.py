"""Helper functions for list and iterables."""

import random as rd
from typing import Sequence, Callable, TypeVar, Optional, Tuple, Iterable, List

# pylint: disable=invalid-name

#: A generic type
T = TypeVar("T")
#: A predicate over :obj:`T` type elements.
PredicateT = Callable[[T], bool]


def split_chunks(seq: Sequence[T], n: int) -> Iterable[List[T]]:
    """Return an iterator with `n` chunks of `seq` elements with approximatly the same size."""
    k, m = divmod(len(seq), n)
    return (seq[i * k + min(i, m) : (i + 1) * k + min(i + 1, m)] for i in range(n))


def find(seq: Sequence[T], pred=PredicateT) -> int:
    """Find the first element of `seq` which holds ``True`` for prdeicate`pred`
    and returns its index or -1 if none found."""
    for i, el in enumerate(seq):
        if pred(el):
            return i
    return -1


def find_all(seq: Sequence[T], pred=PredicateT) -> List[int]:
    """Find all elements of `seq` which hold ``True`` for predicate `pred` and returns their indices."""
    ret = []
    for i, el in enumerate(seq):
        if pred(el):
            ret.append(i)
    return ret


def find_subseq(seq: Sequence[T], sub_seq: Sequence[T]) -> int:
    """Find the first occurence of `sub_seq` in `seq`, if found returns the index of the first element,
    else returns -1."""
    seq = tuple(seq)
    sub_seq = tuple(sub_seq)
    N = len(seq)
    n = len(sub_seq)
    for i in range(N - n + 1):
        if seq[i : i + n] == sub_seq:
            return i
    return -1


def find_subseq_spaced(seq: Sequence[T], sub_seq: Sequence[T]) -> List[int]:
    """If the subsequence `sub_seq` can be derived from `seq` return the indices `sub_seq` elements in `seq`
    else return an empty list.

    A subsequence is derived from a sequence by removing 0 or more elements from the sequence
    without changing the order of the remaining elements.
    If multiple results are possible returns the smallest one in terms of indices.
    """
    ret = []
    sub_seq_it = iter(sub_seq)
    cur_sub_seq = next(sub_seq_it)
    try:
        for i, seq_el in seq:
            if seq_el == cur_sub_seq:
                ret.append(i)
                cur_sub_seq = next(sub_seq_it)
    except StopIteration:
        assert len(ret) == len(sub_seq)
        return ret
    return []


def first_segment_where(
    seq: Sequence[T], pred: Callable[[T], bool], start: int = 0, stop: Optional[int] = None
) -> Tuple[Optional[int], Optional[int]]:
    """Returns the indices of the first segment of `seq` where all elements verify `pred`.
    If no elements verify `pred` returns ``(-1, -1)``.

    Parameters
    ----------
    seq: Sequence[T]
        Sequence of elements
    pred: PredicateT
        Predicate on the elements of `seq`.
    start: int
        Index of the first element to consider.
    stop: int
        Index of the last element to consider plus 1.

    Returns
    -------
        i, j: Tuple[int, int]
            If `i` and `j` are not -1, then ``seq[i]`` is the first element of ``seq[start:stop]``
            for which `pred` returns ``True`` and ``seq[j]`` is the first element of ``seq[i:stop]``
            for which `pred` returns ``False``.
    """
    stop = stop or len(seq)
    beg, end = -1, -1
    for k in range(start, stop):
        if pred(seq[k]):
            if beg == -1:
                beg = k
                end = len(seq)
        else:
            if beg >= 0:
                end = k
                break
    return beg, end


def rd_it(seq: Sequence[T]) -> Iterable[T]:
    """Create a shuffled shallow copy of `seq` and iterate over it."""
    n = len(seq)
    perm = list(range(n))
    rd.shuffle(perm)
    for i in perm:
        yield seq[i]


# class Iterator(object):
#     """Iterator class which cache current value."""

#     def __init__(self, iterator: Iterable[T]):
#         self.iterator: Iterable[T] = iterator
#         self.current: Optional[T] = None

#     def __iter__(self) -> Iterable[T]:
#         return self

#     def __next__(self) -> T:
#         self.current = next(self.iterator)
#         return self.current


if __name__ == "__main__":
    l = "a b c d a b c d e f".split()
    print(l)
    print(find_subseq_spaced(l, "j z".split()))
