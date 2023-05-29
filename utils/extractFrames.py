import cv2
import os

def extractFrames(input_path, output_path):
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
    
    #extractFrames("/Users/jiabowang/Desktop/TFG/mmpose/inputs/padel.mp4", "/Users/jiabowang/Desktop/TFG/mmpose/outputs")