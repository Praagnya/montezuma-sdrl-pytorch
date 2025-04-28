import operator


class SegmentTree:
    def __init__(self, capacity, operation, neutral_element):
        assert capacity > 0 and (capacity & (capacity - 1)) == 0
        self._capacity = capacity
        self._value = [neutral_element for _ in range(2 * capacity)]
        self._operation = operation

    def _reduce(self, start, end, node, node_start, node_end):
        if start == node_start and end == node_end:
            return self._value[node]
        mid = (node_start + node_end) // 2
        if end <= mid:
            return self._reduce(start, end, 2 * node, node_start, mid)
        elif start > mid:
            return self._reduce(start, end, 2 * node + 1, mid + 1, node_end)
        else:
            return self._operation(
                self._reduce(start, mid, 2 * node, node_start, mid),
                self._reduce(mid + 1, end, 2 * node + 1, mid + 1, node_end)
            )

    def reduce(self, start=0, end=None):
        if end is None:
            end = self._capacity - 1
        return self._reduce(start, end, 1, 0, self._capacity - 1)

    def __setitem__(self, idx, val):
        idx += self._capacity
        self._value[idx] = val
        idx //= 2
        while idx >= 1:
            self._value[idx] = self._operation(
                self._value[2 * idx], self._value[2 * idx + 1]
            )
            idx //= 2

    def __getitem__(self, idx):
        return self._value[self._capacity + idx]


class SumSegmentTree(SegmentTree):
    def __init__(self, capacity):
        super().__init__(capacity, operator.add, 0.0)

    def sum(self, start=0, end=None):
        return super().reduce(start, end)

    def find_prefixsum_idx(self, prefixsum):
        assert 0 <= prefixsum <= self.sum() + 1e-5
        idx = 1
        while idx < self._capacity:
            if self._value[2 * idx] > prefixsum:
                idx = 2 * idx
            else:
                prefixsum -= self._value[2 * idx]
                idx = 2 * idx + 1
        return idx - self._capacity


class MinSegmentTree(SegmentTree):
    def __init__(self, capacity):
        super().__init__(capacity, min, float('inf'))

    def min(self, start=0, end=None):
        return super().reduce(start, end)
