# Gesture Detection C++ Demo

This demo showcases inference of Gesture Detection networks using Async API.

## How to Run

./gesture_detection_demo \
-m_d /home/yuxincui/Workspace/open_model_zoo/models/intel/person-detection-asl-0001/FP16/person-detection-asl-0001.xml \
-i /home/yuxincui/Workspace/open_model_zoo/data/classification/image \
-o gesture_detection_image.avi

./gesture_detection_demo \
-m_d /home/yuxincui/Workspace/open_model_zoo/models/intel/person-detection-asl-0001/FP16/person-detection-asl-0001.xml \
-i /home/yuxincui/Workspace/open_model_zoo/data/twsxxn_small.mp4 \
-o gesture_detection_twsxxn.avi

### Supported Models

* person-detection-asl-0001