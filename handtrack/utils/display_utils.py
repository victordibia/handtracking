from handtrack.utils.model_utils import np_array_to_tensor
import matplotlib

matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
from object_detection.utils import visualization_utils as viz_utils
import cv2
import time

label_id_offset = 1


def dispaly_boxes(image_np_with_detections, detections, category_index, min_score=0.3):
    viz_utils.visualize_boxes_and_labels_on_image_array(
        image_np_with_detections,
        detections["detection_boxes"][0].numpy(),
        (detections["detection_classes"][0].numpy() + label_id_offset).astype(int),
        detections["detection_scores"][0].numpy(),
        category_index,
        use_normalized_coordinates=True,
        max_boxes_to_draw=200,
        min_score_thresh=min_score,
        agnostic_mode=False,
        keypoints=None,
        keypoint_scores=None,
        keypoint_edges=None,
    )

    # plt.figure(figsize=(12, 16))
    # plt.imshow(image_np_with_detections)
    # plt.show()


def detect_video(detect_fn, category_index):
    video_source, width, height = 1, 700, 640
    cap = cv2.VideoCapture(video_source)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)

    cv2.namedWindow("Handtrack", cv2.WINDOW_NORMAL)

    while True:
        ret, image_np = cap.read()
        # image_np = cv2.flip(image_np, 1)
        try:
            image_np = cv2.cvtColor(image_np, cv2.COLOR_BGR2RGB)
        except:
            print("Error converting to RGB")

        input_tensor = np_array_to_tensor(image_np)
        start_time = time.time()
        detections, predictions_dict, shapes = detect_fn(input_tensor)
        elapsed_time = time.time() - start_time
        print("fps ", round(1 / elapsed_time, 2))
        dispaly_boxes(image_np, detections, category_index, min_score=0.6)

        cv2.imshow(
            "Handtrack | Press Q to Exit", cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)
        )

        if cv2.waitKey(25) & 0xFF == ord("q"):
            cap.release()
            cv2.destroyAllWindows()
            break
