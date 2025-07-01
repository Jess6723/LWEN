import torch
import torch.nn as nn
import torch.nn.functional as F
from layers.RevIN import RevIN
from layers.FrequencyFilter import FrequencyFilter


class Model(nn.Module):
    def __init__(self, configs):
        super(Model, self).__init__()
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        self.scale = 0.02
        self.revin_layer = RevIN(configs.enc_in, affine=True, subtract_last=False)

        self.hidden_size = configs.hidden_size
        self.split_ratio = configs.split_ratio
        self.hid_ratio = configs.hid_ratio

        self.window_size = configs.window_size
        self.stride = self.window_size // 2

        self.ft_len = (self.window_size // 2) + 1
        self.low_len = int(self.ft_len * self.split_ratio[0])
        self.mid_len = int(self.ft_len * self.split_ratio[1])
        self.high_len = self.ft_len - self.low_len - self.mid_len

        self.high_filter = FrequencyFilter(num_layers=1, freq_len=self.high_len,
                                           hid_len=int(self.high_len * self.hid_ratio))
        self.mid_filter = FrequencyFilter(num_layers=2, freq_len=self.mid_len,
                                          hid_len=int(self.mid_len * self.hid_ratio))
        self.low_filter = FrequencyFilter(num_layers=3, freq_len=self.low_len,
                                          hid_len=int(self.low_len * self.hid_ratio))

        # lernable padding window
        window_type = getattr(torch, f"{configs.window_type}_window", None)
        if window_type is None:
            raise ValueError(f"unknown window type: {configs.window_type}")
        init_window = window_type(self.window_size)
        self.window = nn.Parameter(init_window.clone(), requires_grad=True)

        self.total_len = self.seq_len+self.window_size
        self.num_windows = (self.total_len - self.window_size) // self.stride + 1
        self.embed_size = self.num_windows * self.window_size

        self.fc = nn.Sequential(
            nn.Linear(self.embed_size, self.hidden_size),
            nn.LeakyReLU(),
            nn.Linear(self.hidden_size, self.pred_len)
        )

    def forward(self, x, x_mark_enc, x_dec, x_mark_dec, mask=None):
        B, L, D = x.shape

        # RevIN
        x = self.revin_layer(x, 'norm')  # [B, L, D]
        x = x.permute(0, 2, 1)  # [B, D, L]

        #total_len = max(seq_len, pred_len)
        pad_len = self.total_len - L
        if pad_len > 0:
            left = self.stride  #pad_len // 2
            right = self.stride  #pad_len - left
            x = F.pad(x, (left, right), mode='constant', value=0.0)

        # normalized window function energy
        window = self.window / torch.norm(self.window, p=2)

        outputs = []
        for i in range(self.num_windows):
            start = i * self.stride
            end = start + self.window_size
            x_seg = x[:, :, start:end]             # [B, D, window_size]
            x_win = x_seg * window
            x_fft = torch.fft.rfft(x_win, dim=2)

            # split of frequency domain
            low = x_fft[:, :, :self.low_len]
            mid = x_fft[:, :, self.low_len:self.low_len + self.mid_len]
            high = x_fft[:, :, self.low_len + self.mid_len:]

            low_f = self.low_filter(low)
            mid_f = self.mid_filter(mid)
            high_f = self.high_filter(high)

            # commbine
            x_filtered = torch.cat([low_f, mid_f, high_f], dim=2)
            x_time = torch.fft.irfft(x_filtered, n=self.window_size, dim=2)
            outputs.append(x_time)

        outputs = torch.stack(outputs, dim=1)  # [B, num_windows, D, win_size]
        outputs = outputs.permute(0, 2, 1, 3).reshape(B, D, -1)  # [B, D, embed_size]

        x = self.fc(outputs)  # [B, D, pred_len]
        x = x.permute(0, 2, 1)  # [B, pred_len, D]
        x = self.revin_layer(x, 'denorm')

        return x

    def window_regularization(self):
        """
        window regularization
        """
        l2_loss = torch.mean(self.window ** 2)

        diff = self.window[1:] - self.window[:-1]
        smoothness_loss = torch.mean(diff ** 2)

        return l2_loss + smoothness_loss
