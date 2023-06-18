import os
import cv2

def makeVideo(input_path, output_path):
    """"
    Input:
        - input_path:  ubicación en donde se encuentra las secuencias a generar el vídeo
        - output_path: ubicación en donde se guarda el vídeo generado
    
    Generar vídeos a partir de una secuencia de fotogramas
    """
    os.chdir(input_path)
    output = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'mp4v'), 30, (1920, 1080))
    i = 0
    while(i < 271):
        frame = cv2.imread('frame' + str(i) + '.png') 
        output.write(frame)
        i = i + 1
    cv2.destroyAllWindows()
    output.release()