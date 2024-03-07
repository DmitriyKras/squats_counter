import numpy as np
import onnxruntime
import cv2


class YOLOPoseONNX:
    def __init__(self, onnx_path: str, input_shape: tuple,
                 conf=0.25, iou=0.45):
        """Class for YOLOv8-pose ONNX inference

        Parameters
        ----------
        onnx_path : str
            Path to ONNX file
        input_shape : tuple
            Model's input shape (width, height)
        conf : float
            Confidence YOLO threshold, by default 0.25
        iou : float
            IoU threshold for NMS algo, by default 0.45
        """
        self.onnx_path = onnx_path
        self.input_shape = input_shape
        self.dtype = np.float32
        self.conf = conf
        self.iou = iou
        self.ort_session = onnxruntime.InferenceSession(onnx_path)

    def  __nms(self, boxes: np.ndarray) -> np.ndarray:
        """Perform postprocess and NMS on output boxes

        Parameters
        ----------
        boxes : np.ndarray
            All boxes from YOLO output

        Returns
        -------
        np.ndarray or None
            Filtered boxes
        """
        x = boxes[:, 0]
        y = boxes[:, 1]
        w = boxes[:, 2]
        h = boxes[:, 3]
        conf = boxes[:, 4]
        areas = w * h  # compute areas of boxes
        ordered = conf.argsort()[::-1]  # get sorted indexes of scores in descending order
        keep = []  # boxes to keep
        while ordered.size > 0:
            # Index of the current element:
            i = ordered[0]
            keep.append(i)
            xx1 = np.maximum(x[i], x[ordered[1:]])
            yy1 = np.maximum(y[i], y[ordered[1:]])
            xx2 = np.minimum(x[i] + w[i], x[ordered[1:]] + w[ordered[1:]])
            yy2 = np.minimum(y[i] + h[i], y[ordered[1:]] + h[ordered[1:]])

            width1 = np.maximum(0.0, xx2 - xx1 + 1)
            height1 = np.maximum(0.0, yy2 - yy1 + 1)
            intersection = width1 * height1
            union = (areas[i] + areas[ordered[1:]] - intersection)
            iou = intersection / union
            indexes = np.where(iou <= self.iou)[0]
            ordered = ordered[indexes + 1]

        keep = np.array(keep)
        if len(keep) == 0:
            return None
        boxes = boxes[keep]
        return boxes

    def __call__(self, x:np.ndarray) -> np.ndarray:
        """Perform detection, NMS and kpts scaling

        Parameters

        ----------
        x : np.ndarray
            Plain input frame in BGR (H, W, C) np.uint8

        Returns

        -------
        np.ndarray or None
            Keypoints scaled according to input frame size
        """
        # preprocess
        ratio = (x.shape[1] / self.input_shape[0], x.shape[0] / self.input_shape[1])
        ratio = np.array(ratio)
        x = cv2.resize(x, self.input_shape)
        x = x[:, :, ::-1].transpose(2, 0, 1)[None, ...] / 255.0
        x = x.astype(self.dtype)
        ort_inputs = {self.ort_session.get_inputs()[0].name: x}
        boxes = self.ort_session.run(None, ort_inputs)[0].T.squeeze()
        boxes = boxes[boxes[:, 4] > self.conf, :]  # filter boxes by confidence
        boxes = self.__nms(boxes)  # nms boxes
        if boxes is not None:
            boxes[:, :2] = boxes[:, :2] * ratio
            boxes[:, 2:4] = boxes[:, 2:4] * ratio
            for i in range(17):
                boxes[:, 3*i + 5 : 3*i + 7] = boxes[:, 3*i + 5 : 3*i + 7] * ratio
        return boxes 
    