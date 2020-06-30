import torch
from torch.utils.data import sampler
import numpy as np
import misc as ms

class Random(sampler.Sampler):
    def __init__(self, train_set):
        self.n_samples = len(train_set)

    def __iter__(self):
        indices =  np.random.randint(0, self.n_samples, self.n_samples)
        return iter(torch.from_numpy(indices).long())

    def __len__(self):
        return self.n_samples



class Random10(sampler.Sampler):
    def __init__(self, train_set):
        self.n_samples = len(train_set)
        self.size = min(self.n_samples, 10)
        self.last_indices = None

    def __iter__(self):
        
        indices =  np.random.randint(0, self.n_samples, self.size)

        if self.last_indices is not  None:
            assert not np.in1d(indices, self.last_indices).mean()==1
            
        self.last_indices = indices
        
        return iter(torch.from_numpy(indices).long())

    def __len__(self):
        return self.size


class Sequence(sampler.Sampler):
    def __init__(self, train_set):
        self.n_samples = len(train_set)
        self.size = self.n_samples
        self.last_indices = None

    def __iter__(self):
        indices = np.random.choice(self.n_samples, self.n_samples, replace=False)

        # if self.last_indices is not  None:
        #     assert not np.in1d(indices, self.last_indices).mean()==1
        
        # self.last_indices = indices

        return iter(torch.from_numpy(indices).long())

    def __len__(self):
        return self.size

class Random1000(sampler.Sampler):
    def __init__(self, train_set):
        self.n_samples = len(train_set)
        self.size = min(self.n_samples, 1000)
        self.last_indices = None

    def __iter__(self):
        indices = np.random.randint(0, self.n_samples, self.size)

        if self.last_indices is not  None:
            assert not np.in1d(indices, self.last_indices).mean()==1
        
        self.last_indices = indices

        return iter(torch.from_numpy(indices).long())

    def __len__(self):
        return self.size


class Weighted1000(sampler.Sampler):
    def __init__(self, train_set, replacement=False):
        counts = train_set.counts
        weights = ms.count2weight(counts>0)
        self.weights = torch.DoubleTensor(weights)
        self.num_samples = 1000
        self.replacement = replacement

    def __iter__(self):
        return iter(torch.multinomial(self.weights, self.num_samples, 
                    self.replacement))

    def __len__(self):
        return self.num_samples


class Random3000(sampler.Sampler):
    def __init__(self, train_set):
        self.n_samples = len(train_set)
        self.size = 3000

    def __iter__(self):
        indices =  np.random.randint(0, self.n_samples, self.size)
        return iter(torch.from_numpy(indices).long())

    def __len__(self):
        return self.size