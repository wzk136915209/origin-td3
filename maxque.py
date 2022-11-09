from collections import deque


class MaxQueue:

    def __init__(self, maxlen):
        self.data = deque(maxlen=maxlen)

    def max_value(self):
        # if len(self.data) == 0:
        #     return -1
        # else:
        #     return max(self.data)
        return max(self.data)

    def push_back(self, value):
        self.data.append(value)

    def pop_front(self):
        # if len(self.data) == 0:
        #     return -1
        # else:
        #     return self.data.popleft()
        return self.data.popleft()