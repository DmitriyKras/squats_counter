from yolo_pose.yolo_onnx import YOLOPoseONNX
import cv2
import numpy as np
import time
from utils import utils
from utils.counters import SquatsCounter
import yaml
import argparse
import os


class Squats:
    """
    Base class for pose detection, squats counting and displaying
    """
    def __init__(self, config='config.yaml'):
        """Class for squats counting through the camera

        Parameters
        ----------
        config : str
            Path to yaml config file, by default 'config.yaml'
        """
        # open config file
        with open(config, 'r') as f:
            self.config = yaml.full_load(f)
        self.frame_shape = (self.config['OPTIONS']['frame_width'], self.config['OPTIONS']['frame_height'])
        # initialize yolo model
        self.yolo = YOLOPoseONNX(self.config['MODEL']['onnx_file'], (self.config['MODEL']['width'], self.config['MODEL']['height']),
                           self.config['MODEL']['conf'], self.config['MODEL']['iou'])
        self.cap = cv2.VideoCapture(self.config['OPTIONS']['source'])  # open video or stream
        self.counter = SquatsCounter(self.config['OPTIONS']['leg_angle'], self.config['OPTIONS']['body_angle'])  # initialize squats counter

    def __main_cycle(self, frame: np.ndarray) -> np.ndarray:
        """One iteration of main loop

        Parameters
        ----------
        frame : np.ndarray
            Frame from camera

        Returns
        -------
        np.ndarray
            Processed frame
        """
        font = cv2.FONT_HERSHEY_COMPLEX
        boxes = self.yolo(frame)
        rect_height = int(self.frame_shape[1] / 5)
        rect_width = int(self.frame_shape[0] / 4)
        rect_d = int((self.frame_shape[1] - 4 * rect_height) / 5)  # distance between text strings on frame
        size = 0.6 / 720 * self.frame_shape[1]  # size of text strings
        # angles text
        frame = cv2.putText(frame, 'Angle in knees:', (20, rect_d + int(rect_height * 0.4)), font,
                          size, (0, 255, 0), 1, cv2.LINE_AA)
        frame = cv2.putText(frame, 'Angle in hips:', (20, rect_d + int(rect_height * 0.7)), font,
                          size, (0, 255, 0), 1, cv2.LINE_AA)
        # balanced squats
        frame = cv2.putText(frame, 'Balanced squats:', (20, rect_d * 2 + rect_height + int(rect_height * 0.2)), font,
                          size, (0, 255, 0), 1, cv2.LINE_AA)
        
        if boxes is not None:
            kpts = utils.bboxes2kpts(boxes)[0]  # get kpts from boxes
            # get angles
            leg_angle = utils.leg_angle(kpts)
            body_angle = utils.body_angle(kpts)
            # draw keypoints and skeleton
            frame = utils.draw_kpts(frame, kpts)
            frame = utils.draw_skeleton(frame, kpts)
            # process angles in legs and body
            legs, body = self.counter.step(leg_angle, body_angle)
            color = (0, 0, 255)
            if legs:
                color = (47, 196, 0)
            frame = cv2.putText(frame, str(int(leg_angle)) if not np.isnan(leg_angle) else '180', 
                                (10 + int(0.75 * rect_width), rect_d + int(rect_height * 0.4)), font,
                                size * 1.2, color, 1, cv2.LINE_AA)
            color = (0, 0, 255)
            if body:
                color = (47, 196, 0)
            frame = cv2.putText(frame, str(int(body_angle)) if not np.isnan(body_angle) else '180', 
                                (10 + int(0.75 * rect_width), rect_d + int(rect_height * 0.7)), font,
                                size * 1.2, color, 1, cv2.LINE_AA)
        # get number of counts and put information on frame
        count = self.counter.get_count()
        # show counter
        frame = cv2.putText(frame, str(count), (10 + int(0.75 * rect_width), rect_d * 2 + rect_height + int(rect_height * 0.2)), font,
                          size * 2, (0, 255, 0), 1, cv2.LINE_AA)
        return frame

    def watch(self) -> None:
        """Run project
        """
        fps = 0.0
        tic = 0
        while True:
            ret, frame = self.cap.read()  # get frame
            key = cv2.waitKey(1)
            if not ret:
                print('Reconnecting to camera')
                time.sleep(5)
                self.cap = cv2.VideoCapture(self.config['OPTIONS']['source'])
                continue
            if key == 27:  # ESC key: quit program
                print('Release resources')
                break
            if key == ord(self.config['OPTIONS']['reset_key']):  # q key: reset counter
                self.counter.reset()
            frame = cv2.resize(frame, self.frame_shape)
            frame = self.__main_cycle(frame)  # process frame
            cv2.imshow('SQUATS', frame)  # display frame
            toc = time.time()
            curr_fps = 1.0 / (toc - tic)
            # calculate an exponentially decaying average of fps number
            fps = curr_fps if fps == 0.0 else (fps*0.95 + curr_fps*0.05)
            tic = toc
            print("FPS: ", fps)
        cv2.destroyAllWindows()

    def release(self) -> None:
        """Free resources
        """
        self.cap.release()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--config-file", default="config.yaml",
                        help="path to yaml config file")
    args = parser.parse_args()
    squats = Squats(args.config_file)
    print('Starting...')
    squats.watch()
    squats.release()
            