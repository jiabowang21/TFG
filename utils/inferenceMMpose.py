import os

def inferenceMMpose(input_path, output_path, predictions):
    if (predictions):
        os.system("python /Users/jiabowang/Desktop/TFG/mmpose/demo/topdown_demo_with_mmdet.py \
                   /Users/jiabowang/Desktop/TFG/mmpose/demo/mmdetection_cfg/faster_rcnn_r50_fpn_coco.py \
                   /Users/jiabowang/Desktop/TFG/data/faster_rcnn_r50_fpn_1x_coco_20200130-047c8118.pth \
                   /Users/jiabowang/Desktop/TFG/mmpose/configs/body_2d_keypoint/topdown_heatmap/coco/td-hm_hrnet-w32_8xb64-210e_coco-256x192.py \
                   /Users/jiabowang/Desktop/TFG/data/hrnet_w32_coco_256x192-c78dce93_20200708.pth \
                   --input {} --save-predictions \
                   --output-root {} --device cpu".format(input_path, output_path))
    else:
        os.system("python /Users/jiabowang/Desktop/TFG/mmpose/demo/topdown_demo_with_mmdet.py \
                   /Users/jiabowang/Desktop/TFG/mmpose/demo/mmdetection_cfg/faster_rcnn_r50_fpn_coco.py \
                   /Users/jiabowang/Desktop/TFG/data/faster_rcnn_r50_fpn_1x_coco_20200130-047c8118.pth \
                   /Users/jiabowang/Desktop/TFG/mmpose/configs/body_2d_keypoint/topdown_heatmap/coco/td-hm_hrnet-w32_8xb64-210e_coco-256x192.py \
                   /Users/jiabowang/Desktop/TFG/data/hrnet_w32_coco_256x192-c78dce93_20200708.pth \
                   --input {} \
                   --output-root {} --device cpu".format(input_path, output_path))
        