from PIL import Image
from detectron2.data.transforms import ResizeTransform
from detectron2.data.transforms.augmentation import Augmentation
from numpy.random.mtrand import randint


class RandomResize(Augmentation):
    """ Resize image to a fixed target size"""

    def __init__(self, low, high, interp=Image.BILINEAR):
        """
            low:
            high:
            interp: PIL interpolation method
        """
        self._init(locals())

    def get_transform(self, img):
        new_h = randint(self.low, self.high)
        new_w = randint(self.low, self.high)
        return ResizeTransform(img.shape[0], img.shape[1], new_h, new_w, self.interp)
