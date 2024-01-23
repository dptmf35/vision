from object_detection.utils import label_map_util, visualization_utils as viz_utils
import tensorflow as tf
import cv2
import numpy as np
import time

PBTXT = "./models/label_map.pbtxt"
MODEL = "./models/exported-models-V2/my_model/saved_model/"
resize_ratio = 0.4

detector = tf.saved_model.load(MODEL)
category_index = label_map_util.create_category_index_from_labelmap(PBTXT, True)

cap = cv2.VideoCapture("./licenseplate.mp4")  

try:
    while True:
        ret, img = cap.read()
        if not ret:
            break
        img = cv2.resize(img, (0, 0), fx=resize_ratio, fy=resize_ratio)
        height, width = img.shape[:2]
        img_np = np.array(img)
        start = time.time()
        detections = detector(np.expand_dims(img_np, 0))
        print(f"computing time : {time.time() - start:.2f}")

        num_detections = int(detections.pop('num_detections'))
        detections = {key: value[0, :num_detections].numpy() for key, value in detections.items()}
        detections['num_detections'] = num_detections
        detections['detection_classes'] = detections['detection_classes'].astype(np.int64)

        image_np_with_detections = img_np.copy()
        viz_utils.visualize_boxes_and_labels_on_image_array(
                    image_np_with_detections,
                    detections['detection_boxes'],
                    detections['detection_classes'],
                    detections['detection_scores'],
                    category_index,
                    use_normalized_coordinates=True,
                    # max_boxes_to_draw=5,
                    min_score_thresh=.8,
                    agnostic_mode=False)

        # dict_keys(['raw_detection_boxes', 'detection_anchor_indices', 'detection_classes', 
        # 'raw_detection_scores', 'detection_multiclass_scores', 'detection_scores', 'detection_boxes', 'num_detections'])

        height, width, _ = image_np_with_detections.shape

        for i in range(len(detections['detection_boxes'])):  
            if detections['detection_scores'][i] >= 0.8:
                # get bbox coord
                box = detections['detection_boxes'][i]
                ymin, xmin, ymax, xmax = box

                # normalized -> pixel coord
                (left, right, top, bottom) = (xmin * width, xmax * width, ymin * height, ymax * height)
                (left, right, top, bottom) = map(int, (left, right, top, bottom))
                # print(f"Box {i}: (x1, y1, x2, y2) = ({left}, {top}, {right}, {bottom})")
                ROI = img[top:bottom, left:right]
                blur = cv2.GaussianBlur(ROI, (99, 99), 0)
                img[top:bottom, left:right] = blur


        cv2.imshow("image", img)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
finally:
    cap.release()
    cv2.destroyAllWindows()
