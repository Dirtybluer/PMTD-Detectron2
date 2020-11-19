import io
import os
import logging

from contextlib import redirect_stdout
from fvcore.common.timer import Timer
from fvcore.common.file_io import PathManager
from detectron2.structures import BoxMode
from detectron2.data import DatasetCatalog, MetadataCatalog
from datasets.coco_text.utils import COCO_Text

logger = logging.getLogger(__name__)

# todo: complete the docstrings


def load_coco_text_json(json_file, image_root, mode, subset=None):
    """
    Load a json file storing COCO-Text's annotations.

    :param json_file:
    :param image_root:
    :param mode:
    :param subset:
    :return:
    """
    timer = Timer()
    json_file = PathManager.get_local_path(json_file)

    # omit the original log of COCO-Text api and log the time span independently
    with redirect_stdout(io.StringIO()):
        coco_text = COCO_Text(json_file)
    if timer.seconds() > 1:
        logger.info("Loading {} takes {:.2f} seconds.".format(json_file, timer.seconds()))

    # retrieve image IDs according to the mode and sort indices for reproducible results
    img_ids = coco_text.getImgIds(coco_text.train) if mode == "train" else coco_text.getImgIds(coco_text.train)
    img_ids = sorted(img_ids)
    if subset:
        assert subset <= len(img_ids), \
            "try to retrieve a {} subset of {} images while the total number is {}".format(mode, subset, len(img_ids))
        img_ids = img_ids[:subset]
    # imgs is a list of dicts, each looks something like:
    # {
    #     "id": 540965,
    #     "set": "train",
    #     "width": 640,
    #     "file_name": "COCO_train2014_000000540965.jpg",
    #     "height": 360
    # }
    imgs = coco_text.loadImgs(img_ids)
    ann_ids = [coco_text.imgToAnns[img_id] for img_id in img_ids]
    # anns is a list[list[dict]], where each dict is an annotation
    # record for a text object. The inner list enumerates the text objects in an image
    # and the outer list enumerates over images. Example of anns[0]:
    # [
    #     {
    #         "mask":
    #             [
    #                 468.9,
    #                 286.7,
    #                 468.9,
    #                 295.2,
    #                 ...
    #             ],
    #         "class": "machine printed",
    #         "bbox":
    #             [
    #                 468.9,
    #                 286.7,
    #                 24.1,
    #                 9.1
    #             ],
    #         "image_id": 217925,
    #         "id": 45346,
    #         "language": "english",
    #         "area": 206.06,
    #         "utf8_string": "New",
    #         "legibility": "legible"
    #     },
    #     ...
    # ]
    anns = [coco_text.loadAnns(ann_ids_per_img) for ann_ids_per_img in ann_ids]

    imgs_anns = list(zip(imgs, anns))

    logger.info("Loaded {} images in COCO format from {}".format(len(imgs_anns), json_file))

    dataset_dicts = []

    # construct a dict for each image
    for (img_dict, ann_dict_list) in imgs_anns:
        record = {
            "file_name": os.path.join(image_root, img_dict["file_name"]),
            "height": img_dict["height"],
            "width": img_dict["width"],
            "image_id": img_dict["id"],
        }

        annotations = []
        for ann_dict in ann_dict_list:
            assert ann_dict["image_id"] == record["image_id"]

            annotation = {
                "bbox": ann_dict["bbox"],
                "bbox_mode": BoxMode.XYWH_ABS,
                # Since text is the only class in COCO-Text, category of each annotation should be 0.
                "category_id": 0
            }

            # mask is a list[float] in COCO-Text annotations
            # while a list[list[float]] is required by Detectron2's as standard dataset dicts
            # (see https://detectron2.readthedocs.io/tutorials/datasets.html#standard-dataset-dicts).
            # Thus we wrap the mask with an extra list.
            mask = [ann_dict["mask"]]
            assert len(mask[0]) % 2 == 0
            # mask on an instance(text) is called "segmentation" in Detectron2
            annotation["segmentation"] = mask

            annotations.append(annotation)
        record["annotations"] = annotations
        dataset_dicts.append(record)
    return dataset_dicts


def register_coco_text(json_file, image_root, mode, subset=None, **metadata):
    """
    Register COCO-Text.

    :param json_file:
    :param image_root:
    :param mode:
    :param subset:
    :return:
    """
    assert isinstance(json_file, (str, os.PathLike)), json_file
    assert isinstance(image_root, (str, os.PathLike)), image_root
    assert mode in {"train", "val"}, "mode should be either train or val"
    assert isinstance(subset, (int, type(None))), "subset should be an integer while get {}".format(type(subset))

    DatasetCatalog.register("COCO-Text", lambda: load_coco_text_json(json_file, image_root, mode, subset))
    MetadataCatalog.get("COCO-Text").set(json_file=json_file, image_root=image_root, **metadata)


if __name__ == "__main__":
    """
    Test the COCO-Text json dataset loader
    
    Usage:
        python -m detectron2.data.datasets.coco \
            path/to/json path/to/image_root mode subset
    """
    import sys
    dicts = load_coco_text_json(sys.argv[1], sys.argv[2], sys.argv[3], int(sys.argv[4]))
    print(dicts[0])
