import os

def inferenceMMpose(input_path, output_path, predictions, bbox):
    """"
    Input:
        - input_path:  ubicación en donde se encuentra la imagen a inferir
        - output_path: ubicación en donde se guarda la imagen inferida
        - predictions: parámetro para indicar si se quiere guardar la información 
        - bbox:        parámetro para indicar si se quiere dibujar los cuadros delimitadores sobre la imagen
    
    Inferir la pose y el cuadro delimitador de la imagen de entrada
    """
    if (predictions):
        if (bbox):
            os.system("python ./mmpose/demo/topdown_demo_with_mmdet.py \
                       ./mmpose/demo/mmdetection_cfg/faster_rcnn_r50_fpn_coco.py \
                       ./data/faster_rcnn_r50_fpn_1x_coco_20200130-047c8118.pth \
                       ./mmpose/configs/body_2d_keypoint/topdown_heatmap/coco/td-hm_hrnet-w32_8xb64-210e_coco-256x192.py \
                       ./data/hrnet_w32_coco_256x192-c78dce93_20200708.pth \
                       --input {} --save-predictions --draw-bbox \
                       --output-root {} --device cpu".format(input_path, output_path))
        else:
            os.system("python ./mmpose/demo/topdown_demo_with_mmdet.py \
                       ./mmpose/demo/mmdetection_cfg/faster_rcnn_r50_fpn_coco.py \
                       ./data/faster_rcnn_r50_fpn_1x_coco_20200130-047c8118.pth \
                       ./mmpose/configs/body_2d_keypoint/topdown_heatmap/coco/td-hm_hrnet-w32_8xb64-210e_coco-256x192.py \
                       ./data/hrnet_w32_coco_256x192-c78dce93_20200708.pth \
                       --input {} --save-predictions \
                       --output-root {} --device cpu".format(input_path, output_path))
    else:
        if (bbox):
            os.system("python ./mmpose/demo/topdown_demo_with_mmdet.py \
                       ./mmpose/demo/mmdetection_cfg/faster_rcnn_r50_fpn_coco.py \
                       ./data/faster_rcnn_r50_fpn_1x_coco_20200130-047c8118.pth \
                       ./mmpose/configs/body_2d_keypoint/topdown_heatmap/coco/td-hm_hrnet-w32_8xb64-210e_coco-256x192.py \
                       ./data/hrnet_w32_coco_256x192-c78dce93_20200708.pth \
                       --input {} --draw-bbox \
                       --output-root {} --device cpu".format(input_path, output_path))
        else:
            os.system("python ./mmpose/demo/topdown_demo_with_mmdet.py \
                       ./mmpose/demo/mmdetection_cfg/faster_rcnn_r50_fpn_coco.py \
                       ./data/faster_rcnn_r50_fpn_1x_coco_20200130-047c8118.pth \
                       ./mmpose/configs/body_2d_keypoint/topdown_heatmap/coco/td-hm_hrnet-w32_8xb64-210e_coco-256x192.py \
                       ./data/hrnet_w32_coco_256x192-c78dce93_20200708.pth \
                       --input {} \
                       --output-root {} --device cpu".format(input_path, output_path))
