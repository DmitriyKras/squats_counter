import cv2
import numpy as np


def _compute_angle(pt1: np.ndarray, pt2: np.ndarray, ver: np.ndarray) -> float:
    """Compute angle between pt1 and pt2 relative to vertex ver

    Parameters
    ----------
    pt1 : np.ndarray
        Array of x, y, score
    pt2 : np.ndarray
        Array of x, y, score
    ver : np.ndarray
        Array of x, y, score

    Returns
    -------
    float
        Angle in degrees
    """
    cosine = np.dot(pt1[:2] - ver[:2], pt2[:2] - ver[:2]) / (np.linalg.norm(pt1[:2] - ver[:2]) * np.linalg.norm(pt2[:2] - ver[:2]))
    angle = np.degrees(np.arccos(cosine))
    return angle


def leg_angle(kpts: dict) -> float:
    """Compute angle between ankle and hip

    Parameters
    ----------
    kpts : dict
        Dictionary with keypoints

    Returns
    -------
    float
        Mean angle in degrees
    """
    left_angle = _compute_angle(kpts['left_ankle'], kpts['left_hip'], kpts['left_knee'])
    right_angle = _compute_angle(kpts['right_ankle'], kpts['right_hip'], kpts['right_knee'])
    return (left_angle + right_angle) / 2


def body_angle(kpts: dict) -> float:
    """Compute angle between body and hip

    Parameters
    ----------
    kpts : dict
        Dictionary with keypoints

    Returns
    -------
    float
        Mean angle in degrees
    """
    left_angle = _compute_angle(kpts['left_shoulder'], kpts['left_knee'], kpts['left_hip'])
    right_angle = _compute_angle(kpts['right_shoulder'], kpts['right_knee'], kpts['right_hip'])
    return (left_angle + right_angle) / 2


def draw_kpts(frame: np.ndarray, kpts: dict, color=(0, 0, 255)) -> np.ndarray:
    """Draw keypoints on given frame

    Parameters
    ----------
    frame : np.ndarray
        Frame to draw keypoints
    kpts : dict
        Dictionary with keypoints
    color : tuple
        Color of points

    Returns
    -------
    np.ndarray
        Frame with keypoints
    """
    for _, pt in kpts.items():
        frame = cv2.circle(frame, (pt[0], pt[1]), 2, color, -1)
    return frame


def draw_skeleton(frame: np.ndarray, kpts: dict, color=(0, 156, 255)) -> np.ndarray:
    """Draw keypoints on given frame

    Parameters
    ----------
    frame : np.ndarray
        Frame to draw keypoints
    kpts : dict
        Dictionary with keypoints
    color : tuple
        Color of points

    Returns
    -------
    np.ndarray
        Frame with keypoints
    """
    frame = cv2.line(frame, (kpts['left_ankle'][0], kpts['left_ankle'][1]), (kpts['left_knee'][0], kpts['left_knee'][1]), color, 1)
    frame = cv2.line(frame, (kpts['left_hip'][0], kpts['left_hip'][1]), (kpts['left_knee'][0], kpts['left_knee'][1]), color, 1)
    frame = cv2.line(frame, (kpts['left_hip'][0], kpts['left_hip'][1]), (kpts['right_hip'][0], kpts['right_hip'][1]), color, 1)
    frame = cv2.line(frame, (kpts['right_knee'][0], kpts['right_knee'][1]), (kpts['right_hip'][0], kpts['right_hip'][1]), color, 1)
    frame = cv2.line(frame, (kpts['right_knee'][0], kpts['right_knee'][1]), (kpts['right_ankle'][0], kpts['right_ankle'][1]), color, 1)
    frame = cv2.line(frame, (kpts['right_hip'][0], kpts['right_hip'][1]), (kpts['right_shoulder'][0], kpts['right_shoulder'][1]), color, 1)
    frame = cv2.line(frame, (kpts['left_hip'][0], kpts['left_hip'][1]), (kpts['left_shoulder'][0], kpts['left_shoulder'][1]), color, 1)
    frame = cv2.line(frame, (kpts['right_shoulder'][0], kpts['right_shoulder'][1]), (kpts['left_shoulder'][0], kpts['left_shoulder'][1]), color, 1)
    frame = cv2.line(frame, (kpts['right_shoulder'][0], kpts['right_shoulder'][1]), (kpts['right_elbow'][0], kpts['right_elbow'][1]), color, 1)
    frame = cv2.line(frame, (kpts['right_wrist'][0], kpts['right_wrist'][1]), (kpts['right_elbow'][0], kpts['right_elbow'][1]), color, 1)
    frame = cv2.line(frame, (kpts['left_shoulder'][0], kpts['left_shoulder'][1]), (kpts['left_elbow'][0], kpts['left_elbow'][1]), color, 1)
    frame = cv2.line(frame, (kpts['left_wrist'][0], kpts['left_wrist'][1]), (kpts['left_elbow'][0], kpts['left_elbow'][1]), color, 1)
    return frame


def bboxes2kpts(bboxes: np.ndarray) -> list:
    """Convert bboxes from YOLO output
    to dictionary with keypoints

    Parameters
    ----------
    bboxes : np.ndarray
        Bboxes from YOLO output

    Returns
    -------
    list
        List with dictionaries of keypoints
    """
    kpts = []
    for box in bboxes[:, 5:]:
        kpts.append({
            # nose
            'nose': np.array((box[0], box[1])).astype(int),
            # left shoulder
            'left_shoulder': np.array((box[15], box[16])).astype(int),
            # right shoulder
            'right_shoulder': np.array((box[18], box[19])).astype(int),
            # left elbow
            'left_elbow': np.array((box[21], box[22])).astype(int),
            # right elbow
            'right_elbow': np.array((box[24], box[25])).astype(int),
            # left wrist
            'left_wrist': np.array((box[27], box[28])).astype(int),
            # right wrist
            'right_wrist': np.array((box[30], box[31])).astype(int),
            # left hip
            'left_hip': np.array((box[33], box[34])).astype(int),
            # right hip
            'right_hip': np.array((box[36], box[37])).astype(int),
            # left knee
            'left_knee': np.array((box[39], box[40])).astype(int),
            # right knee
            'right_knee': np.array((box[42], box[43])).astype(int),
            # left ankle
            'left_ankle': np.array((box[45], box[46])).astype(int),
            # right ankle
            'right_ankle': np.array((box[48], box[49])).astype(int)
        })
    return kpts
