import copy
import torch
import logging
import numpy as np
from typing import List, Union
from detectron2.data import detection_utils as utils
from detectron2.data import transforms as T


class DatasetMapper:
    """
    A callable which takes a dataset dict in Detectron2 Dataset format,
    and map it into a format used by the model.

    The callable currently does the following:

    1. Read the image from "file_name"
    2. Applies transforms to the image and annotations
    3. Prepare data and annotations to Tensor and :class:`Instances`
    """

    def __init__(
            self,
            is_train: bool,
            *,
            augmentations: List[Union[T.Augmentation, T.Transform]],
            image_format: str,
            use_instance_mask: bool = False,
            instance_mask_format: str = "polygon",
            recompute_boxes: bool = False
    ):
        """
        NOTE: this interface is experimental.

        Args:
            is_train: whether it's used in training or inference
            augmentations: a list of augmentations or deterministic transforms to apply
            image_format: an image format supported by :func:`detection_utils.read_image`.
            use_instance_mask: whether to process instance segmentation annotations, if available
            recompute_boxes: whether to overwrite bounding box annotations
                by computing tight bounding boxes from instance mask annotations.
        """
        if recompute_boxes:
            assert use_instance_mask, "recompute_boxes requires instance masks"
        # fmt: off
        self.is_train = is_train
        self.augmentations = augmentations
        self.image_format = image_format
        self.use_instance_mask = use_instance_mask
        self.instance_mask_format = instance_mask_format
        self.recompute_boxes = recompute_boxes
        # fmt: on
        logger = logging.getLogger(__name__)
        logger.info("Augmentations used in training: " + str(augmentations))

    def __call__(self, dataset_dict):
        """
        Args:
            dataset_dict (dict): Metadata of one image, in Detectron2 Dataset format.

        Returns:
            dict: a format that builtin models in detectron2 accept
        """
        dataset_dict = copy.deepcopy(dataset_dict)
        # read an image into BGR format and do sanity check on image size
        image = utils.read_image(dataset_dict["file_name"], format="BGR")
        utils.check_image_size(dataset_dict, image)

        # apply transformations on the image
        aug_input = T.StandardAugInput(image)
        transforms = aug_input.apply_augmentations(self.augmentations)
        image = aug_input.image

        # PIL reads an image with channel order as H x W x C,
        # we have to transpose the order to C x H x W for PyTorch
        # and convert the image from numpy array to tensor
        image_shape = image.shape[:2]
        dataset_dict["image"] = torch.as_tensor(np.ascontiguousarray(image.transpose(2, 0, 1)))

        if not self.is_train:
            dataset_dict.pop("annotations", None)
            return dataset_dict

        if "annotations" in dataset_dict:
            # apply transformations on annotations
            anns = [
                utils.transform_instance_annotations(obj, transforms, image_shape)
                for obj in dataset_dict.pop("annotations")
            ]
            # convert annotations into Instances()
            instances = utils.annotations_to_instances(anns, image_shape, mask_format=self.instance_mask_format)

            # recompute the bbox with the corresponding mask
            if self.recompute_boxes:
                instances.gt_boxes = instances.gt_masks.get_bounding_boxes()
            dataset_dict["instances"] = utils.filter_empty_instances(instances)
        return dataset_dict
