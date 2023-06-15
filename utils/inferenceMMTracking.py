import mmcv
import tempfile
import numpy as np
from mmtrack.apis import inference_mot, init_model

def inferencia(mot_config, input_video, output):
    """"
    Input:
        - mot_config:      configuración del modelo 
        - input_video:     ubicación del video de entrada
        - output:          ubicación donde se quiere guardar el video inferido
    Output:
        - track:           lista con el seguimiento de cada jugador
    """
    imgs = mmcv.VideoReader(input_video)
    # build the model from a config file
    mot_model = init_model(mot_config, device='cpu')
    prog_bar = mmcv.ProgressBar(len(imgs))
    out_dir = tempfile.TemporaryDirectory()
    out_path = out_dir.name
    
    #límites del campo
    top_left_x = 140.
    top_left_y = 100.
    
    bottom_right_x = 1740.
    bottom_right_y = 930.
    
    track = dict()
    
    for i, img in enumerate(imgs):
        
        result = inference_mot(mot_model, img, frame_id=i)
        
        if len(result.get('det_bboxes')[0] > 0):
            detections = result.get('det_bboxes')[0]
            for k in range(0, len(detections)):
                if detections[k][0] < top_left_x or detections[k][1] < top_left_y:
                    result['det_bboxes'][0][k][4] = -1.
                elif detections[k][2] > bottom_right_x or detections[k][3] > bottom_right_y:
                    result['det_bboxes'][0][k][4] = -1.
        
        if len(result.get('track_bboxes')[0]) > 0:
            bboxes = result.get('track_bboxes')[0]
            for k in range(0, len(bboxes)):
                if bboxes[k][1] < top_left_x or bboxes[k][2] < top_left_y:
                    result['track_bboxes'][0][k][5] = -1.
                elif bboxes[k][3] > bottom_right_x or bboxes[k][4] > bottom_right_y:
                    result['track_bboxes'][0][k][5] = -1.
                else:
                    if (bboxes[k][0] not in track):
                        track[bboxes[k][0]] = list()
                    track[bboxes[k][0]].append(np.append(bboxes[k][1:], i))

        mot_model.show_result(
                img, 
                result,
                show=False,
                score_thr=0.,
                thickness=6,
                wait_time=0,
                out_file=f'{out_path}/{i:06d}.jpg')
        prog_bar.update()
    
    print(f'\n making the output video at {output} with a FPS of {imgs.fps}')
    mmcv.frames2video(out_path, output, fps=imgs.fps, fourcc='mp4v')
    out_dir.cleanup()
    return track
