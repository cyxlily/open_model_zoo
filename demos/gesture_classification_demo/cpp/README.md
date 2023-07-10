# Gesture Classification C++ Demo

This demo showcases inference of Gesture Classification networks using Async API.

## How to Run

./gesture_classification_demo -m_a /home/yuxincui/Workspace/open_model_zoo/models/intel/common-sign-language-0002/FP16/common-sign-language-0002.xml \
-i /home/yuxincui/Workspace/open_model_zoo/data/classification/image \
-c /home/yuxincui/Workspace/open_model_zoo/data/dataset_classes/common_sign_language12.txt \
--loop

./gesture_classification_demo -m_a /home/yuxincui/Workspace/open_model_zoo/models/intel/common-sign-language-0002/FP16/common-sign-language-0002.xml \
-i /home/yuxincui/Workspace/open_model_zoo/data/twsxxn_small.mp4 \
-o gesture_classification_twsxxn.avi \
-c /home/yuxincui/Workspace/open_model_zoo/data/dataset_classes/common_sign_language12.txt

### Supported Models

* common-sign-language-0002
