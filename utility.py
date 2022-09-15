import numpy as np


class Utility:
    # leg order: LF-LH-RF-RH
    # def __init__(self):
    def to_pin(self, input, offset):
        return np.concatenate([input[0: offset], input[offset + 3: offset + 6], input[offset + 9: offset + 12],
                               input[offset + 0: offset + 3], input[offset + 6: offset + 9]])

    def from_pin(self, input, offset):
        return np.concatenate([input[0: offset], input[offset + 6: offset + 9], input[offset + 0: offset + 3],
                               input[offset + 9: offset + 12],
                               input[offset + 3: offset + 6]])

    def m2l(self, input):
        return np.array(input)[:, 0].tolist()
