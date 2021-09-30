import logging

import numpy as np
from handtrack.utils.model_utils import (
    get_detect_fn,
    get_label_map,
    img_to_nparray,
    np_array_to_tensor,
)

from handtrack.utils.display_utils import dispaly_boxes, detect_video


class HandModel:
    def __init__(self, model_size):
        self.load_model(model_size)

    def load_model(self, model_size):
        logging.info(" >> loading model with size", model_size)
        self.size = model_size
        self.detect_fn = get_detect_fn(model_size=model_size, ckpt_num="ckpt-12")
        (
            self.categories,
            self.category_index,
            self.label_map_dict,
        ) = get_label_map()

    def detect(self, image_path="", input_mode="image"):
        print("detection")
        if input_mode == "image":
            image_np = img_to_nparray(image_path)
            input_tensor = np_array_to_tensor(image_np)
            detections, predictions_dict, shapes = self.detect_fn(input_tensor)
            dispaly_boxes(
                image_np.copy(), detections, self.category_index, min_score=0.7
            )
        elif input_mode == "video":
            detect_video(self.detect_fn, self.category_index)
