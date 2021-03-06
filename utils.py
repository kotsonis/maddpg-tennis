import numpy as np
import torch 
import operator
class SegmentTree():
    def __init__(self, capacity, operation, neutral_element):
        """Build a Segment Tree data structure.
        https://en.wikipedia.org/wiki/Segment_tree
        Arguments
        ---------
        capacity: 
            Total size of the array - must be a power of two.
        operation: lambda obj, obj -> obj
            an operation for combining elements (eg. sum, max)
        neutral_element: 
            neutral element for the operation above. eg. float('-inf')
            for max and 0 for sum.
        """
        assert capacity > 0 and capacity & (capacity - 1) == 0, "capacity must be positive and a power of 2."
        self._capacity = capacity
        self._value = np.full(2*capacity,neutral_element, dtype=float)
        self._operation = operation

    def _reduce_helper(self, start: int, end:int, node:int, node_start:int, node_end:int):
        """ recursively apply `self.operation` to a tree subset with sequential representation """
        if start == node_start and end == node_end:
            return self._value[node]
        mid = (node_start + node_end) // 2
        if end <= mid:
            return self._reduce_helper(start, end, 2 * node, node_start, mid)
        else:
            if mid + 1 <= start:
                return self._reduce_helper(start, end, 2 * node + 1, mid + 1, node_end)
            else:
                return self._operation(
                    self._reduce_helper(start, mid, 2 * node, node_start, mid),
                    self._reduce_helper(mid + 1, end, 2 * node + 1, mid + 1, node_end)
                )

    def reduce(self, start:int =0, end:int =None):
        """Returns result of applying `self.operation` to a contiguous subsequence of the array.
            self.operation(arr[start], operation(arr[start+1], operation(... arr[end])))
        Returns
        -------
        reduced: obj
            result of reducing self.operation over the specified range of array elements.
        """
        if end is None:
            end = self._capacity
        if end < 0:
            end += self._capacity
        end -= 1
        return self._reduce_helper(start, end, 1, 0, self._capacity - 1)

    def __setitem__(self, idx:int, val:float):
        """ set value for a node in tree and update parents """

        idx += self._capacity   
        self._value[idx] = val
        
        idx //= 2       ## go to parent
        
        while idx >= 1:
            self._value[idx] = self._operation(
                self._value[2 * idx],
                self._value[2 * idx + 1]
            )
            idx //= 2   ## keep moving to parent in b-tree
        

    def __getitem__(self, idx):
        """ get value for item in tree """
        assert 0 <= idx < self._capacity
        return self._value[self._capacity + idx]


class SumSegmentTree(SegmentTree):
    """ Segment Tree with cummulative sum being the segmentation operator """
    def __init__(self, capacity):
        """ initialize SumSegmentTree as a SegmentTree with operation being addition and neutral value being 0 """
        super(SumSegmentTree, self).__init__(capacity=capacity,operation=operator.add,neutral_element=0.0)

    def sum(self, start:int =0, end:int =None) :
        """Returns sum of elements from index `start` to index `end` """
        return super(SumSegmentTree, self).reduce(start, end) # run the reduce operator on the SegmentTree

    def find_prefixsum_idx(self, prefixsum:float):
        """Find the highest index `i` in the SumSegmentTree such that sum(data[0] ... data[`i - i`]) is less than `prefixsum`
        Arguments:
        ----
        prefixsum:
            the cummulative sum that we are querying for
        Returns:
        ----
        idx:
            highest index satisfying the prefixsum constraint
        """
        assert 0 <= prefixsum <= self.sum() + 1e-5, "prefixsum outside current data boundaries"
        idx = 1     # tree root node index
        while idx < self._capacity: # stay within b-tree structure and not the data themselves
            if self._value[2 * idx] > prefixsum: # if left child is greater, move to that child and keep drilling
                idx = 2 * idx 
            else:
                prefixsum -= self._value[2 * idx] # if right child is greater, subtract left child sum and move to right to keep drilling
                idx = 2 * idx + 1
        return idx - self._capacity 


class MinSegmentTree(SegmentTree):
    """ Segment Tree segmented on the minimum value operator """
    def __init__(self, capacity:int):
        super(MinSegmentTree, self).__init__( # initialize as a SegmentTree with operation being min comparison and neutral value being infinity
            capacity=capacity,
            operation=min,
            neutral_element=float('inf')
        )

    def min(self, start=0, end=None):
        """Returns minimum value of Tree elements from index `start` to index `end` """
        return super(MinSegmentTree, self).reduce(start, end)

#  Ornstein-Uhlenbeck Noise class
# from https://github.com/songrotek/DDPG/blob/master/ou_noise.py
class OUNoise:

    def __init__(self, action_dimension, scale=1.0, mu=0, theta=0.15, sigma=0.2):
        self.action_dimension = action_dimension
        self.scale = scale
        self.mu = mu
        self.theta = theta
        self.sigma = sigma
        self.state = np.ones(self.action_dimension) * self.mu
        self.reset()

    def reset(self):
        self.state = np.ones(self.action_dimension) * self.mu

    def noise(self):
        x = self.state
        dx = self.theta * (self.mu - x) + self.sigma * np.random.randn(len(x))
        self.state = x + dx
        return torch.tensor(self.state * self.scale).float()

def hidden_init(layer):
    fan_in = layer.weight.data.size()[0]
    lim = 1. / np.sqrt(fan_in)
    return (-lim, lim)
