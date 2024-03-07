# squats_counter

Pet project for counting number of squats exercises. Squat detection based on keypoints obtained from YOLOv8 onnx inference and angles in knees and hips computation.

# Project overview

`yolo_pose/yolo_onnx.py` contains class for YOLOv8 onnx inference. `utils/utils.py` and `utils/counters.py` contain utils for drawing and converting keypoints and squats counter class. `test_yolo.py` - script for testing YOLOv8 inference on test image. `squats.py` - main script to run project.

# Project running

Run project with:

`python3 squats.py`

Press `q` to reset counter, press `Esc` to quit.