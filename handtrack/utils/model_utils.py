import tensorflow as tf
from PIL import Image, ImageDraw, ImageFont
from six import BytesIO
import numpy as np

from object_detection.utils import label_map_util
from object_detection.utils import config_util
from object_detection.utils import visualization_utils as viz_utils
from object_detection.builders import model_builder
import os

root_file_path = os.path.dirname(os.path.abspath(__file__))


def get_model_detection_function(model):
    """Get a tf.function for detection."""

    @tf.function
    def detect_fn(image):
        """Detect objects in image."""

        image, shapes = model.preprocess(image)
        prediction_dict = model.predict(image, shapes)
        detections = model.postprocess(prediction_dict, shapes)

        return detections, prediction_dict, tf.reshape(shapes, [-1])

    return detect_fn


def img_to_nparray(img_path):
    img = tf.io.gfile.GFile(img_path, "rb").read()
    image = Image.open(BytesIO(img))
    (im_width, im_height) = image.size
    return np.array(image.getdata()).reshape((im_height, im_width, 3)).astype(np.uint8)


def np_array_to_tensor(image_np):
    return tf.convert_to_tensor(np.expand_dims(image_np, 0), dtype=tf.float32)


def get_model_details(model_size):
    if model_size == "small":

        model_dir = os.path.join(root_file_path, "..", "checkpoints", model_size)
        pipeline_config = os.path.join(model_dir, "pipeline.config")
        print(model_dir, pipeline_config)
        return model_dir, pipeline_config


def get_detect_fn(model_size="small", ckpt_num="ckpt-8"):

    model_dir, pipeline_config = get_model_details(model_size)
    # Load pipeline config and build a detection model
    configs = config_util.get_configs_from_pipeline_file(pipeline_config)
    model_config = configs["model"]
    detection_model = model_builder.build(model_config=model_config, is_training=False)

    # Restore checkpoint
    ckpt = tf.compat.v2.train.Checkpoint(model=detection_model)
    ckpt.restore(os.path.join(model_dir, ckpt_num)).expect_partial()
    detect_fn = get_model_detection_function(detection_model)
    return detect_fn


def get_label_map():
    label_map_path = os.path.join(root_file_path, "..", "checkpoints", "label_map.txt")
    label_map = label_map_util.load_labelmap(label_map_path)
    categories = label_map_util.convert_label_map_to_categories(
        label_map,
        max_num_classes=label_map_util.get_max_label_map_index(label_map),
        use_display_name=True,
    )
    category_index = label_map_util.create_category_index(categories)
    label_map_dict = label_map_util.get_label_map_dict(
        label_map, use_display_name=False
    )
    return categories, category_index, label_map_dict
