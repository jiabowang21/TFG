import os

def inferenceMMTracking(input_path, output_path):
    os.system("python /Users/jiabowang/Desktop/TFG/mmtracking/demo/demo_mot_vis.py \
    /Users/jiabowang/Desktop/TFG/mmtracking/configs/mot/deepsort/deepsort_faster-rcnn_fpn_4e_mot17-private-half.py \
    --input {} \
    --output {} --device cpu".format(input_path, output_path))
