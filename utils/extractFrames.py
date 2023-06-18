import cv2
import os

def extractFrames(input_path, output_path):
    """"
    Input:
        - input_path:  ubicación en donde se encuentra el vídeo a fragmentar
        - output_path: ubicación en donde se guardan el conjunto de fotogramas extraídos
    
    Extraer el conjunto de fotogramas de un vídeo 
    """
    if not os.path.exists(output_path):
        # If it doesn't exist, create it
        os.makedirs(output_path)
    
    os.chdir(output_path)

    video = cv2.VideoCapture(input_path)
    currentframe = 0
    while True:
        ret, frame = video.read()
        if ret:   #mientras haya frames que extraer
            name = 'frame' + str(currentframe) + '.png'
            cv2.imwrite(name, frame)
            currentframe = currentframe + 1
        else:
            break
    video.release()
    cv2.destroyAllWindows()