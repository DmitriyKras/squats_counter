import cv2
from utils.utils import draw_skeleton, bboxes2kpts, draw_kpts
from yolo_pose.yolo_onnx import YOLOPoseONNX


yolo = YOLOPoseONNX('yolo_pose/weights/yolov8n-pose.onnx', (512, 320), conf=0.5)

img = cv2.imread('yolo_pose/weights/test.jpg')
boxes = yolo(img)
kpts = bboxes2kpts(boxes)
for box, kpt in zip(boxes[:, :4], kpts):
    img = cv2.rectangle(img, (int(box[0] - box[2] / 2), int(box[1] - box[3] / 2)), (int(box[0] + box[2] / 2), int(box[1] + box[3] / 2)), (0, 0, 255), 2)
    img = draw_skeleton(img, kpt)
    img = draw_kpts(img, kpt)

cv2.imshow('test', img)
cv2.waitKey(0)
cv2.destroyAllWindows()
