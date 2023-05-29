import json
import torch
import cv2
from utils.datos import transformar_a_coordenadas_hip, deshacer_normalizacion, cargar_datos_blender, tratamiento_datos, cambiar_input_frames

def inferenciaPosicion(model, path):
    data = json.load(open(path))
    X = []
    hip = []
    n = len(data['instance_info'])
    for i in range(0, n):
        k = data['instance_info'][i]['keypoints']
        X.append(k)
        hip1 = k[11]
        hip2 = k[12]
        hip.append([(k[11][0] + k[12][0])/2, (k[11][1] + k[12][1])/2])
    
    for i in range(0, len(X)):
        for j in range(0, len(X[i])):
            X[i][j] = transformar_a_coordenadas_hip(hip[i], X[i][j])    

    X = torch.tensor(X, dtype=torch.float32)
    X = torch.reshape(X, (len(X), X.shape[1]*X.shape[2]))
    predicciones = model(X)
    predicciones = predicciones.tolist()

    for i in range(0, len(predicciones)):
        predicciones[i] = deshacer_normalizacion(predicciones[i], hip[i])
    
    return predicciones

def dibujar_prediccion(prediccion, input_path):
    imagen = cv2.imread(input_path)
    for i in range(0, len(prediccion)):
        imagen = cv2.circle(imagen, (int(prediccion[i][0]), int(prediccion[i][1])), radius = 6, color = (0, 0, 255), thickness=-1)
    return imagen

def inferenciaVideo(model_path, path_keypoints, path_truth, video_input, video_output):
    X, y, hip = cargar_datos_blender(path_keypoints, path_truth)
    X, y = tratamiento_datos(X, y, hip)
    # convertirlos en tensores
    X = torch.tensor(X, dtype=torch.float32)
    X = torch.reshape(X, (X.shape[0], X.shape[1]*X.shape[2]))
    y = torch.tensor(y, dtype=torch.float32)

    model = torch.load(model_path)
    p = model(X)
    p = p.tolist()

    for i in range(0, len(p)):
        p[i] = deshacer_normalizacion(p[i], hip[i])

    cap = cv2.VideoCapture(video_input)

    fps = cap.get(cv2.CAP_PROP_FPS)

    output = cv2.VideoWriter(video_output, cv2.VideoWriter_fourcc(*'mp4v'), fps, (1920, 1080))
    i = 0

    while(True and i < len(p)):
        ret, frame = cap.read()
        frame = cv2.circle(frame, (int(p[i][0]),int(p[i][1])), radius=6, color=(0, 0, 255), thickness=-1)
        output.write(frame)
        i = i + 1

    cv2.destroyAllWindows()
    output.release()
    cap.release()
    
def inferenciaVideoMultiploFrames(model_path, path_keypoints, path_truth, video_input, video_output, frames, offset):
    X_aux, y_aux, hip = cargar_datos_blender(path_keypoints, path_truth)

    X_aux, y_aux = tratamiento_datos(X_aux, y_aux, hip)
    
    X, y = cambiar_input_frames(X_aux, y_aux, frames, offset)
    
    X = torch.tensor(X, dtype=torch.float32)
    X = torch.reshape(X, (X.shape[0], X.shape[1]*X.shape[2]*X.shape[3]))
    
    model = torch.load(model_path)
    p = model(X)
    p = p.tolist()
    
    secuencia = int((offset/2)*frames)
    for i in range(0, len(p)):
        p[i] = deshacer_normalizacion(p[i], hip[i+secuencia])
        
    cap = cv2.VideoCapture(video_input)

    fps = cap.get(cv2.CAP_PROP_FPS)

    output = cv2.VideoWriter(video_output, cv2.VideoWriter_fourcc(*'mp4v'), fps, (1920, 1080))
    i = 0

    while(True and i < len(p)):
        ret, frame = cap.read()
        if (i >= secuencia and i < len(p)):
            frame = cv2.circle(frame, (int(p[i-secuencia][0]),int(p[i-secuencia][1])), radius=6, color=(0, 0, 255), thickness=-1)
        output.write(frame)
        i = i + 1

    cv2.destroyAllWindows()
    output.release()
    cap.release()
        
    