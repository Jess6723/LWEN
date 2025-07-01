import numpy as np
import torch
import torch.nn as nn

class ComplexLinear(nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        self.real_linear = nn.Linear(in_features, out_features)
        self.imag_linear = nn.Linear(in_features, out_features)

    def forward(self, x):
        real_part = self.real_linear(x.real) - self.imag_linear(x.imag)
        imag_part = self.real_linear(x.imag) + self.imag_linear(x.real)
        return torch.complex(real_part, imag_part)

class ComplexRelu(nn.Module):
    def __init__(self, real_bias=0.0, imag_bias=0.0):
        super().__init__()
        self.real_bias = nn.Parameter(torch.tensor(real_bias, dtype=torch.float32))
        self.imag_bias = nn.Parameter(torch.tensor(imag_bias, dtype=torch.float32))

    def forward(self, x):
        real_part = torch.relu(x.real) + self.real_bias
        imag_part = torch.relu(x.imag) + self.imag_bias
        return torch.complex(real_part, imag_part)


class FrequencyFilter(nn.Module):
    def __init__(self, num_layers: int, freq_len: int, hid_len: int):
        super().__init__()

        if num_layers < 1:
            raise ValueError("num_layers don't less than 1")

        layers = []

        if num_layers == 1:
            # freq_len -> freq_len
            layers.append(ComplexLinear(freq_len, freq_len))
        else:
            # freq_len -> hid_len -> ... -> hid_len -> freq_len
            layers.append(ComplexLinear(freq_len, hid_len))
            layers.append(ComplexRelu())

            # hid_len -> hid_len
            for _ in range(num_layers - 2):
                layers.append(ComplexLinear(hid_len, hid_len))
                layers.append(ComplexRelu())

            # hid_len -> freq_len（no activation）
            layers.append(ComplexLinear(hid_len, freq_len))
        self.filters = nn.Sequential(*layers)

    def forward(self, x):
        return self.filters(x)
