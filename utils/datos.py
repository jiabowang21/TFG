import json

def cargar_datos_blender(path_keypoints, path_truth):
    """"
    Input:
        - path_keypoints: string 
        - path_truth:     string
    Output:
        - X: lista que contiene las coordenadas de los keypoints de todos los frames
        - y: lista que contiene las coordenadas del ground truth de todos los frames
        - hip: lista que contiene las coordenadas centrales de los hip de los jugadores
    Se obtienen las coordenadas de los keypoints con sus respectivos ground truth
    """
    keypoints = json.load(open(path_keypoints))
    no_keypoints = []    # frames que no se han detectado keypoints
    hip = []             # coordenadas centrales de los hips
    X = []               # coordenadas de los keypoints de cada frame
    for i in range(0, len(keypoints['instance_info'])):
        if len(keypoints['instance_info'][i]['instances']) > 0:
            X.append(keypoints['instance_info'][i]['instances'][0]['keypoints'])
            hip1 = keypoints['instance_info'][i]['instances'][0]['keypoints'][11]
            hip2 = keypoints['instance_info'][i]['instances'][0]['keypoints'][12]
            hip.append([(hip1[0] + hip2[0])/2, (hip1[1] + hip2[1])/2])
        else:
            no_keypoints.append(i)
    y = list()           # coordenadas de las posiciones de cada frame (ground truth)
    with open(path_truth,'r') as fp: 
        for line in fp.readlines(): 
            col = line.strip().split(";") 
            new_item = [float(col[1]),float(col[2])] 
            y.append(new_item) 
    y = y[0:14660]    

    # suprimir los frames de los cuales no se han podido detectar keypoints
    no_keypoints.sort(reverse = True)
    for i in no_keypoints:
        y.pop(i)
    
    return X, y, hip

def transformar_a_coordenadas_hip(coordenada, punto):
    """"
    Input:
        - coordenada: [float, float] 
        - punto:      [float, float]
    Output:
        - transformada: [float, float]
    Se ha obtenido las coordenadas del vector "punto" respecto a la base de "coordenada"
    """
    punto1 = punto[0] - coordenada[0]
    punto2 = punto[1] - coordenada[1]
    transformada = [punto1, punto2]
    return transformada

def deshacer_normalizacion(punto, coordenada):
    punto1 = punto[0] + coordenada[0]
    punto2 = punto[1] + coordenada[1]
    return [punto1, punto2]

def tratamiento_datos(X, y, hip):
    """"
    Input:
        - X: lista que contiene las coordenadas de los keypoints de todos los frames
        - y: lista que contiene las coordenadas del ground truth de todos los frames
        - hip: lista que contiene las coordenadas centrales de los hip de los jugadores
    Output:
        - X: lista que contiene las coordenadas de los keypoints de todos los frames respecto a las coordenadas hip
        - y: lista que contiene las coordenadas del ground truth de todos los frames respecto a las coordenadas hip
    Se cambia la base las coordenadas de los keypoints y de sus respectivos ground truth
    """
    for i in range(0, len(X)):
        for j in range(0, len(X[i])):
            X[i][j] = transformar_a_coordenadas_hip(hip[i], X[i][j])
    
    for i in range(0, len(y)):
        y[i] = transformar_a_coordenadas_hip(hip[i], y[i])
    
    return X, y

def cambiar_input_frames(X_aux, y_aux, frames, offset):
    X = []
    y = []
    secuencia = int((offset/2)*frames)
    for i in range(secuencia, len(X_aux) - secuencia):
        aux_x = []
        for j in range(i-secuencia, i+secuencia, offset):
            aux_x.append(X_aux[j])
        X.append(aux_x)
        y.append(y_aux[i])
    
    return X, y
    
    
    
    
    
    