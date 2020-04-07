import random as rd
from typing import Sequence, Callable, TypeVar, Generator, Optional, Tuple, Iterable, Generic, List


T = TypeVar("T")
PredicateT = Callable[[T], bool]

def split_chunks(seq: Sequence[T], n:int) -> Iterable[List[T]]:
    """Return an iterator with `n` chunks of `seq` elements with approximatly the same size."""
    k, m = divmod(len(seq), n)
    return (seq[i * k + min(i, m):(i + 1) * k + min(i + 1, m)] for i in range(n))

def find(seq: Sequence[T], pred=PredicateT) -> int:
    for i, el in enumerate(seq):
        if pred(el): return i
    return -1


def find_all(seq: Sequence[T], pred=PredicateT) -> List[int]:
    ret = []
    for i, el in enumerate(seq):
        if pred(el): ret.append(i)
    return ret


def find_subseq(seq: Sequence[T], sub_seq: Sequence[T]) -> int:
    """Find the first occurence of sub_seq in seq, if found returns the index of the first element, else returns -1."""
    N = len(seq)
    n = len(sub_seq)
    for i in range(N - n + 1):
        if seq[i : i+n] == sub_seq:
            return i
    return -1

def first_segment_where(seq: Sequence[T], pred: Callable[[T], bool],
                        start:int=0, stop:Optional[int]=None) -> Tuple[Optional[int], Optional[int]]:
    """Returns the indexes of the first segment of `seq` where every element verify `pred`. If no elements verify `pred` returns -1, -1.
    
    Args
    ----
        seq: Sequence[T]
            Sequence of elements
        pred: Callable[T, bool]
            Predicate on the elements of `seq`.

    Returns
    -------
        i, j: Tuple[int, int] or Tuple[None, None]
            if i and j are not -1, then seq[i] is the first element of `seq` for which pred(seq[i]) = True 
            and seq[j] is the first element of seq[i:] for which pred(seq[j]) = False.
    """
    stop = stop or len(seq)
    beg, end = -1, -1
    for k in range(start, stop):
        if pred(seq[k]) == True:
            if beg == -1:
                beg = k
                end = len(seq)
        else:
            if beg >= 0:
                end = k
                break
    return beg, end


def rd_it(seq: Sequence[T]) -> Generator[T, None, None]:
    """Randomly iterate over the elements of `seq`."""
    n = len(seq)
    perm = list(range(n))
    rd.shuffle(perm)
    for i in perm:
        yield seq[i]


class Iterator(object):
    """Iterator class which cache current value, accessible in `current` attribute."""
    def __init__(self, iterator: Iterable[T]):
        self.iterator: Iterable[T] = iterator
        self.current: Optional[T] = None
    
    def __iter__(self) -> Iterable[T]:
        return self

    def __next__(self) -> T:
        self.current = next(self.iterator)
        return self.current


if __name__ == "__main__":
    l = "A B C D E F".split()
    print(l)
    assert find_subseq(l, "C D".split()) == 2
    assert find_subseq(l, "D C".split()) == -1
    assert find_subseq(l, "A B C".split()) == 0
    assert find_subseq(l, "F".split()) == 5

    print(find_subseq(["NP-SUJ", "VN", "NP-ATS"], ["NP-SUJ", "VN", "NP-ATS"]))

    import random as rd
    l = rd.choices([3, 6, 1], k=4)
    i, j = first_segment_where(l, lambda el: el % 3 == 0)
    print("i, j:", i, j)
    print("".join(map(str, l)))
    if i is not None and j is not None:
        print(" "*(i) + "[" + " "*(j-i-1) + ")")
