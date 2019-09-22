"""
A python wrapper for the darknet components of target_finder_model
"""
import target_finder_model as tfm
import numpy as np
import cv2


class DarknetModel:

    def __init__(self, weights_fn=None, config_fn=None,
                 classes=None, cpu=True):

        self.classes = classes

        # Init model
        self.net = cv2.dnn.readNetFromDarknet(config_fn, weights_fn)
        self.net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)

        if cpu:
            self.net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)

        # Locate output layers
        layers = self.net.getLayerNames()
        self.out_layers = [layers[i[0] - 1]
                           for i in self.net.getUnconnectedOutLayers()]


class Yolo3Detector(DarknetModel):

    def __init__(self, *args, **kwargs):
        kwargs['weights_fn'] = kwargs.get('weights_fn', tfm.yolo3_weights)
        kwargs['config_fn'] = kwargs.get('config_fn', tfm.yolo3_file)
        kwargs['classes'] = tfm.YOLO_CLASSES
        super().__init__(*args, **kwargs)

    def _filter_nms(self, classes, confs, boxes, thresh):
        detects = []
        best_idxs = cv2.dnn.NMSBoxes(boxes, confs, 0.05, thresh)
        for i_ in best_idxs:
            i = i_[0]
            detects.append((classes[i], confs[i], boxes[i]))
        return detects

    def detect_all(self, images, threshold=0.05, nms_thresh=.40):

        if len(images) == 0:
            return []

        elif len(images) == 1:
            # OpenCV Darknet doesnt like lonely inputs
            images = [images[0], np.copy(images[0])]

        h, w, _ = images[0].shape
        n = len(images)
        num_classes = len(self.classes)

        detections = []

        blob = cv2.dnn.blobFromImages(images, 1 / 255, (h, w), [0, 0, 0], 1)
        self.net.setInput(blob)
        net_out = self.net.forward(self.out_layers)

        for k in range(n):

            # storing box/conf/classes seperately for NMS
            shape_boxes = []
            shape_confidences = []
            shape_classes = []

            alpha_boxes = []
            alpha_confidences = []
            alpha_classes = []

            local_detects = []

            for out_layer in net_out:

                out_tensor = out_layer[k]

                for detection in out_tensor:

                    scores = detection[5:]
                    class_idx = np.argmax(scores)
                    conf = float(scores[class_idx])

                    if conf > threshold and class_idx < num_classes:

                        center_x = int(detection[0] * w)
                        center_y = int(detection[1] * h)
                        width = int(detection[2] * w)
                        height = int(detection[3] * h)
                        left = int(center_x - width / 2)
                        top = int(center_y - height / 2)

                        class_name = self.classes[class_idx]
                        dims = [left, top, width, height]

                        # place alpha and shape bboxes in diff sets
                        if len(class_name) != 1:
                            shape_classes.append(class_name)
                            shape_confidences.append(conf)
                            shape_boxes.append(dims)
                        else:
                            alpha_classes.append(class_name)
                            alpha_confidences.append(conf)
                            alpha_boxes.append(dims)

            local_detects.extend(self._filter_nms(shape_classes,
                                                  shape_confidences,
                                                  shape_boxes,
                                                  nms_thresh))
            local_detects.extend(self._filter_nms(alpha_classes,
                                                  alpha_confidences,
                                                  alpha_boxes,
                                                  nms_thresh))

            detections.append(local_detects)

        return detections


class PreClassifier(DarknetModel):

    def __init__(self, *args, **kwargs):
        kwargs['weights_fn'] = kwargs.get('weights_fn', tfm.preclf_weights)
        kwargs['config_fn'] = kwargs.get('config_fn', tfm.preclf_file)
        kwargs['classes'] = tfm.CLF_CLASSES
        super().__init__(*args, **kwargs)

    def classify_all(self, images):

        h, w, _ = images[0].shape

        blob = cv2.dnn.blobFromImages(images, 1 / 255, (h, w), [0, 0, 0], 1)
        self.net.setInput(blob)

        net_out = self.net.forward(self.out_layers)
        prediction = np.squeeze(net_out)

        return [self.classes[np.argmax(pred)] for pred in prediction]
