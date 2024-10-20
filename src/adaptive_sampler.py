from torch.utils.data import Sampler
import random


class AdaptiveSampler(Sampler):
    def __init__(self, dataset, batch_size, total_epochs):
        self.dataset = dataset
        self.batch_size = batch_size
        self.total_epochs = total_epochs
        self.epoch = 0

    def set_epoch(self, epoch):
        self.epoch = epoch

    def __iter__(self):
        indices = list(range(len(self.dataset)))
        random.shuffle(indices)

        keep_prob = (self.epoch) / self.total_epochs

        filtered_indices = []
        for idx in indices:
            if self.dataset[idx]["gts"].argmax(dim=0).sum() > 0:
                filtered_indices.append(idx)
            elif random.random() < keep_prob:
                filtered_indices.append(idx)

        # Ensure we have enough samples to form full batches
        while len(filtered_indices) % self.batch_size != 0:
            filtered_indices.append(random.choice(indices))

        return iter(filtered_indices)

    def __len__(self):
        # This ensures that we always return full batches
        return len(self.dataset)
