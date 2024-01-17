import torch
import torch.utils.data as data


class syth_data(data.Dataset):
    def __init__(self, n_sample, n_dim=30):
        self.n_sample = n_sample
        self.data = torch.zeros([n_sample, n_dim])
        self.center = 1 * torch.zeros([2, n_dim])
        self.center[0, 0] = 1
        self.center[1, -1] = 1

        for i in range(n_sample):
            if i % 2 == 0:
                self.data[i, :] = self.center[0, :] + 0.01 * torch.rand([n_dim])
            else:
                self.data[i, :] = self.center[1, :] + 0.01 * torch.rand([n_dim])

    def __getitem__(self, index):
        return self.data[index, :]

    def __len__(self):
        return self.n_sample
