from handtrack.handmodel import handmodel
from handtrack import HandModel

hand_model = HandModel(model_size="small")
# hand_model.detect(image_path="handtrack/utils/test_image.jpeg")
hand_model.detect(input_mode="video")