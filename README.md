# Pose detection + Position estimation + Player Tracking 
## Pose detection
Para la detección de la pose de un jugador de pádel, se usa el enfoque de arriba hacia abajo. Utiliza un modelo de detección (Faster R-CNN) y un modelo de estimación de la pose humana (HRNet). La implementación de este sistema se basa en MMPose, un framework de código abierto de OpenMMLab. 
En el notebook DetectionDEMO.ipynb se muestra un ejemplo de inferencia del modelo con imágenes de jugadores de pádel durante un partido.

## Position Estimation
Para la estimación de la posición sobre la pista de los jugadores, se ha diseñado una red neuronal completamente conectada. Hay entrenados dos modelos, uno que trata cada fotograma de manera independiente (data/Models) y el otro que trata un conjunto de 23 fotogramas repartidos en una secuencia de 138 fotogramas (data/Models2).
En el notebook PositionEstimationDEMO.ipynb se muestra los diferentes pasos del diseño y entrenamiento del modelo, también hay ejemplos de inferencia con imágenes reales. 

## Player Tracking
Finalmente, para el seguimiento de los jugadores durante un partido, se ha utilizado el modelo Tracktor. A la vez, la implementación se basa en MMTracking, otro framework de código abierto de OpenMMLab.
En el notebook PositionEstimationDEMO.ipynb se muestra un ejemplo de inferencia.
