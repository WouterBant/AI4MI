from batchgenerators.dataloading.data_loader import SlimDataLoaderBase
from batchgenerators.dataloading.multi_threaded_augmenter import MultiThreadedAugmenter
import numpy as np
from torch.utils.data import Dataset, DataLoader
import torch
from batchgenerators.dataloading.data_loader import SlimDataLoaderBase


class ImageDataset(SlimDataLoaderBase):
    def __init__(self, images, batch_size):
        super(ImageDataset, self).__init__(None, batch_size)
        self.images = [
            self.add_channel_dim(img) for img in images
        ]  # Keep original size
        self.images = [img.astype(np.float32) for img in self.images]

    def add_channel_dim(self, img):
        # Ensure image has a channel dimension (1, H, W)
        return np.expand_dims(img, axis=0)  # Add channel dimension (1, H, W)

    def generate_train_batch(self):
        # Randomly select indices for the batch
        indices = np.random.choice(len(self.images), self.batch_size, replace=False)
        batch_data = np.stack([self.images[i] for i in indices], axis=0)
        return {"data": batch_data}


class CTImageDataset(SlimDataLoaderBase):
    """
    Custom DataLoader to handle paired CT images and ground truth masks for medical image processing.
    """

    def __init__(self, images, gts, batch_size):
        super(CTImageDataset, self).__init__(None, batch_size)
        self.images = [
            self.add_channel_dim(img) for img in images
        ]  # Add channel dimension
        self.gts = [
            self.add_channel_dim(gt) for gt in gts
        ]  # Add channel dimension to GT
        self.images = [
            img.astype(np.float32) for img in self.images
        ]  # Ensure float32 for consistency
        self.gts = [
            gt.astype(np.float32) for gt in self.gts
        ]  # Ensure float32 for GT as well

    def add_channel_dim(self, img):
        return np.expand_dims(img, axis=0)  # Add channel dimension (1, H, W)

    def generate_train_batch(self):
        # Randomly select indices for the batch
        indices = np.random.choice(len(self.images), self.batch_size, replace=False)
        batch_data = np.stack([self.images[i] for i in indices], axis=0)
        batch_gts = np.stack([self.gts[i] for i in indices], axis=0)
        return {"data": batch_data, "gt": batch_gts}  # Return both input data and GT


class DummyDL(SlimDataLoaderBase):
    def __init__(self, num_threads_in_mt=8):
        super(DummyDL, self).__init__(None, None, num_threads_in_mt)
        self._data = list(range(100))
        self.current_position = 0
        self.was_initialized = False

    def reset(self):
        self.current_position = self.thread_id
        self.was_initialized = True

    def generate_train_batch(self):
        if not self.was_initialized:
            self.reset()
        idx = self.current_position
        if idx < len(self._data):
            self.current_position = idx + self.number_of_threads_in_multithreaded
            return self._data[idx]
        else:
            self.reset()
            raise StopIteration


class DummyDLWithShuffle(DummyDL):
    def __init__(self, num_threads_in_mt=8):
        super(DummyDLWithShuffle, self).__init__(num_threads_in_mt)
        self.num_restarted = 0
        self.data_order = np.arange(len(self._data))

    def reset(self):
        super(DummyDLWithShuffle, self).reset()
        rs = np.random.RandomState(self.num_restarted)
        rs.shuffle(self.data_order)
        self.num_restarted = self.num_restarted + 1

    def generate_train_batch(self):
        if not self.was_initialized:
            self.reset()
        idx = self.current_position
        if idx < len(self._data):
            self.current_position = idx + self.number_of_threads_in_multithreaded
            return self._data[self.data_order[idx]]
        else:
            self.reset()
            raise StopIteration


if __name__ == "__main__":
    """
    Why is is so hard to iterate only once over my entire training dataset when MultiThreadedAugmenter is used?
    This is because MultiThreadedAugmenter will spawn num_threads workers and each worker will hold a copy of the entire
    pipeline, including the DataLoader. Therefore, if your DataLoader is configured to run over the training data once, but 
    you have 8 threads then what you will be getting from the MultiThreadedAugmenter is an iteration over eight times your 
    training dataset"""

    """
    HELP I want to iterate over all my training data once per epoch.
    Say no more. We go your back. Here is a simple example how you can do that.

    We create a dummy dataloader that has the numbers of 0 to 99 in its _data variable. In the MultiThreadedAugmenter, each 
    DataLoader will know what thread ID it has. We use that information to iterate over the training data. Since there are 
    3 threads, each individual dataloader must return every third item (and start in a different position)
    """

    dl = DummyDL(num_threads_in_mt=3)
    mt = MultiThreadedAugmenter(dl, None, 3, 1, None)

    for i in mt:
        print(i)

    """
    You can run the mt as often as you want because the DataLoader it will reset itself before raising StopIteration
    """
    for i in mt:
        print(i)

    for i in mt:
        print(i)

    """
    But wait. Isn't it suboptimal to iterate over training data always in the same order? Correct. Try this:
    """

    dl = DummyDLWithShuffle(num_threads_in_mt=3)
    mt = MultiThreadedAugmenter(dl, None, 3, 1, None)

    batches = []
    for i in mt:
        batches.append(i)
    print(batches)
    assert (
        len(np.unique(batches)) == 100 and len(batches) == 100
    )  # assert makes sure we got what we wanted

    """
    Once again you can run that as often as you want
    """

    batches = []
    for i in mt:
        batches.append(i)
    print(batches)
    assert len(np.unique(batches)) == 100 and len(batches) == 100

    batches = []
    for i in mt:
        batches.append(i)
    print(batches)
    assert len(np.unique(batches)) == 100 and len(batches) == 100
