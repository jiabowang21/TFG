import os
import cv2

def makeVideo(input_path, output_path):
    os.chdir(input_path)
    output = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'mp4v'), 30, (1920, 1080))
    i = 0
    while(i < 271):
        frame = cv2.imread('frame' + str(i) + '.png') 
        output.write(frame)
        i = i + 1
    cv2.destroyAllWindows()
    output.release()