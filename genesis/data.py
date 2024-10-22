import numpy as np
import gzip
import struct
from .autograd import Tensor

from typing import Iterator, Optional, List, Sized, Union, Iterable, Any


class Transform:
    def __call__(self, x):
        raise NotImplementedError


class RandomFlipHorizontal(Transform):
    def __init__(self, p = 0.5):
        self.p = p

    def __call__(self, img):
        """
        Horizonally flip an image, specified as n H x W x C NDArray.
        Args:
            img: H x W x C NDArray of an image
        Returns:
            H x W x C ndarray corresponding to image flipped with probability self.p
        Note: use the provided code to provide randomness, for easier testing
        """
        flip_img = np.random.rand() < self.p
        if flip_img:
            img = img[:, ::-1, :]
        return img


class RandomCrop(Transform):
    def __init__(self, padding=3):
        self.padding = padding

    def __call__(self, img):
        """ Zero pad and then randomly crop an image.
        Args:
             img: H x W x C NDArray of an image
        Return 
            H x W x C NAArray of cliped image
        Note: generate the image shifted by shift_x, shift_y specified below
        """
        shift_x, shift_y = np.random.randint(low=-self.padding, high=self.padding+1, size=2)
        h, w, c = img.shape
        new_img = np.zeros([h + self.padding * 2, w + self.padding * 2, c])
        new_img[self.padding: self.padding + h, self.padding: self.padding + w, :] = img
        coord_x = self.padding + shift_x
        coord_y = self.padding + shift_y
        return new_img[coord_x: coord_x + h, coord_y: coord_y + w, :]


class Dataset:
    r"""An abstract class representing a `Dataset`.

    All subclasses should overwrite :meth:`__getitem__`, supporting fetching a
    data sample for a given key. Subclasses must also overwrite
    :meth:`__len__`, which is expected to return the size of the dataset.
    """

    def __init__(self, transforms: Optional[List] = None):
        self.transforms = transforms

    def __getitem__(self, index) -> object:
        raise NotImplementedError

    def __len__(self) -> int:
        raise NotImplementedError

    def apply_transforms(self, x):
        if self.transforms is not None:
            # apply the transforms
            for tform in self.transforms:
                x = tform(x)
        return x


class DataLoader:
    r"""
    Data loader. Combines a dataset and a sampler, and provides an iterable over
    the given dataset.

    Args:
        dataset (Dataset): dataset from which to load the data.

        batch_size (int, optional): how many samples per batch to load (default: ``1``).

        shuffle (bool, optional): set to ``True`` to have the data reshuffled at every epoch (default: ``False``).
    """
    dataset: Dataset
    batch_size: Optional[int]

    def __init__(
        self,
        dataset: Dataset,
        batch_size: Optional[int] = 1,
        shuffle: bool = False,
        device=None,
    ):
        self.dataset = dataset
        self.shuffle = shuffle
        self.batch_size = batch_size
        self.device = device
        if not self.shuffle:
            self.ordering = np.array_split(
                    np.arange(len(dataset)), range(batch_size, len(dataset), batch_size))

    def __iter__(self):
        self.idx = 0
        self.len = len(self.dataset) // self.batch_size
        if self.shuffle: 
            tmp_range = np.arange(len(self.dataset))
            np.random.shuffle(tmp_range)
            self.ordering = np.array_split(tmp_range, range(self.batch_size, len(self.dataset), self.batch_size))
        return self

    def __next__(self):
        if self.idx < len(self.ordering):
            data = self.dataset[self.ordering[self.idx]]
            self.idx += 1
            return [Tensor(x, device=self.device) for x in data]
        else:
            raise StopIteration


class MNISTDataset(Dataset):
    def __init__(
        self,
        image_filename: str,
        label_filename: str,
        transforms: Optional[List] = None,
    ):
        with gzip.open(image_filename, "rb") as f:
            self.X = np.frombuffer(f.read(), np.uint8, offset=16).reshape(-1, 784).astype('float32') / 255
        with gzip.open(label_filename, "rb") as f:
            self.y = np.frombuffer(f.read(), np.uint8, offset=8)
        self.transforms = transforms

    def __getitem__(self, index) -> object:
        x = self.X[index]
        if len(x.shape) > 1:
            for i in range(x.shape[0]):
                xx = x[i].reshape(28, 28, 1)
                if self.transforms is not None:
                    for tform in self.transforms:
                        xx = tform(xx)
                x[i] = xx.reshape(-1)
            return (x, self.y[index])
        else:
            x = x.reshape(28, 28, 1)
            if self.transforms is not None:
                for tform in self.transforms:
                    x = tform(x)
            return (x, self.y[index])

    def __len__(self) -> int:
        return len(self.X)

class NDArrayDataset(Dataset):
    def __init__(self, *arrays):
        self.arrays = arrays

    def __len__(self) -> int:
        return self.arrays[0].shape[0]

    def __getitem__(self, i) -> object:
        return tuple([a[i] for a in self.arrays])

