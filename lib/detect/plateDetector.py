import numpy as np
import sys
import collections
import tensorflow as tf

sys.path.append('..')

from ..detect.utils import label_map_util
from ..detect.utils import visualization_utils as vis_util
from ..detect.utils import label_map_util
from ..detect.object_dt_image import *
# Path to label map file
PATH_TO_LABELS = './model/license_plate_ssd_15_11/labelmap-1-license_plate.pbtxt'
PATH_TO_CKPT = './model/license_plate_ssd_15_11/frozen_inference_graph.pb'
# Load the Tensorflow model into memory.
detection_graph_full = tf.Graph()
with detection_graph_full.as_default():
    od_graph_def = tf.GraphDef()
    with tf.gfile.GFile(PATH_TO_CKPT, 'rb') as fid:
        serialized_graph = fid.read()
        od_graph_def.ParseFromString(serialized_graph)
        tf.import_graph_def(od_graph_def, name='')
    sess = tf.Session(graph=detection_graph_full)
NUM_CLASSES = 1

# Load the label map.
# Label maps map indices to category names, so that when our convolution
# network predicts `5`, we know that this corresponds to `king`.
# Here we use internal utility functions, but anything that returns a
# dictionary mapping integers to appropriate string labels would be fine
label_map = label_map_util.load_labelmap(PATH_TO_LABELS)
categories = label_map_util.convert_label_map_to_categories(label_map, max_num_classes=NUM_CLASSES, use_display_name=True)
category_index = label_map_util.create_category_index(categories)

def get_license_plate_full(image, detection_graph_full):
    box_to_display_str_map = collections.defaultdict(list)
    min_score_thresh = .5
    max_boxes_to_draw = 20
    im_height, im_width, _ = image.shape
    #image_np = load_image_into_numpy_array(image)
    # get the matrixs of result
    #output_dict = run_inference_for_single_image(image, detection_graph_full)

    # Input tensor is the image
    image_tensor = detection_graph_full.get_tensor_by_name('image_tensor:0')

    # Output tensors are the detection boxes, scores, and classes
    # Each box represents a part of the image where a particular object was detected
    detection_boxes = detection_graph_full.get_tensor_by_name('detection_boxes:0')

    # Each score represents level of confidence for each of the objects.
    # The score is shown on the result image, together with the class label.
    detection_scores = detection_graph_full.get_tensor_by_name('detection_scores:0')
    detection_classes = detection_graph_full.get_tensor_by_name('detection_classes:0')

    # Number of objects detected
    num_detections = detection_graph_full.get_tensor_by_name('num_detections:0')

    # Load image using OpenCV and
    # expand image dimensions to have shape: [1, None, None, 3]
    # i.e. a single-column array, where each item in the column has the pixel RGB value
    #image = cv2.imread(PATH_TO_IMAGE)
    image_expanded = np.expand_dims(image, axis=0)

    # Perform the actual detection by running the model with the image as input
    (boxes, scores, classes, num) = sess.run(
        [detection_boxes, detection_scores, detection_classes, num_detections],
        feed_dict={image_tensor: image_expanded})
    boxes = np.squeeze(boxes)
    scores = np.squeeze(scores)
    classes = np.squeeze(classes).astype(np.int32)
    # print(output_dict)
    for i in range(min(max_boxes_to_draw, boxes.shape[0])):
        #If class detection is not human, continue
        if scores is None or scores[i] > min_score_thresh:
            box = tuple(boxes[i].tolist())
            ymin, xmin, ymax, xmax = box
            # left = xmin, right = xmax, top = ymin, bottom = ymax
            coods = (xmin * im_width, ymin * im_height,
                     xmax * im_width, ymax * im_height)
            display_str = ''
            if classes[i] in category_index.keys():
                class_name = category_index[classes[i]]['name']
            else:
                class_name = 'N/A'
            #display_str = str(class_name)

            coods = list(map(int, coods))
            #remove
            #object_image = preprocess_image(image, coods)
            #until here
            box_to_display_str_map[tuple(coods)].append('{}'.format(class_name)) #remove brand name
            # print(box_to_display_str_map)
    return box_to_display_str_map