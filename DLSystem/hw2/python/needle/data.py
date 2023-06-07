import numpy as np
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
        img = np.pad(img, ((self.padding, self.padding), (self.padding, self.padding), (0, 0)), mode='constant')
        img = img[self.padding+shift_x:self.padding+shift_x+img.shape[0]-2*self.padding,\
            self.padding+shift_y:self.padding+shift_y+img.shape[1]-2*self.padding,:]
        return img


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
        batch_size (int, optional): how many samples per batch to load
            (default: ``1``).
        shuffle (bool, optional): set to ``True`` to have the data reshuffled
            at every epoch (default: ``False``).
     """
    dataset: Dataset
    batch_size: Optional[int]

    def __init__(
        self,
        dataset: Dataset,
        batch_size: Optional[int] = 1,
        shuffle: bool = False,
    ):

        self.dataset = dataset
        self.shuffle = shuffle
        self.batch_size = batch_size
        if not self.shuffle:
            self.ordering = np.array_split(np.arange(len(dataset)), 
                                           range(batch_size, len(dataset), batch_size))

    def __iter__(self):
        self.index = 0
        if self.shuffle:
            indexes = np.arange(len(self.dataset))
            np.random.shuffle(indexes)
            self.ordering = np.array_split(indexes, 
                                           range(self.batch_size, len(self.dataset), self.batch_size))
        return self

    def __next__(self):
        if self.index == len(self.ordering):
            raise StopIteration()
        samples = [Tensor(x) for x in self.dataset[self.ordering[self.index]]]
        self.index += 1
        return samples


class MNISTDataset(Dataset):
    def __init__(
        self,
        image_filename: str,
        label_filename: str,
        transforms: Optional[List] = None,
    ):
        super().__init__(transforms)
        self.images, self.labels = self.parse_mnist(image_filename, label_filename)

    def parse_mnist(self,   image_filesname, label_filename):

        import gzip
        import struct
        with gzip.open(image_filesname, "rb") as f:
            magic_num, nums, rows, cols = struct.unpack('>IIII', f.read(16))
            images = np.frombuffer(f.read(), dtype=np.uint8) \
                .reshape(nums, rows * cols).astype(np.float32)
            image_max = np.max(images)
            image_min = np.min(images)
            images = (images-image_min)/(image_max-image_min)
        
        with gzip.open(label_filename, "rb") as f:
            magic_num, nums = struct.unpack('>II', f.read(8))
            labels = np.frombuffer(f.read(), dtype=np.uint8)
        
        images = images.astype(np.float32)
        images = images.reshape((-1, 28, 28, 1))

        return (images, labels)

    def __getitem__(self, index) -> object:
        image, label = self.images[index], self.labels[index]
        # if self.transforms:
        #     for transform in self.transforms:
        #         image = transform(image)
        image = self.apply_transforms(image)

        return image, label

    def __len__(self) -> int:
        return self.images.shape[0]

class NDArrayDataset(Dataset):
    def __init__(self, *arrays):
        self.arrays = arrays

    def __len__(self) -> int:
        return self.arrays[0].shape[0]

    def __getitem__(self, i) -> object:
        return tuple([a[i] for a in self.arrays])
