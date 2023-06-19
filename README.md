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

## Instalación
1. Instalar PyTorch: \
  conda install pytorch torchvision -c pytorch (GPU) \
  conda install pytorch torchvision cpuonly -c pytorch (CPU) \
2. Instalar MMEgine \
  pip install -U openmim \
  mim install mmengine \

Para usar el modelo de detección de la pose:
1. Instalar MMCV:
  mim install "mmcv>=2.0.0"
2. Instalar MMDetection:
  mim install "mmdet>=3.0.0"
3. Instalar MMPose: \
  cd mmpose \
  pip install -r requirements.txt \
  pip install -v -e . \

Para usar el modelo de seguimiento:
1. Instalar MMCV:
  mim install "mmcv<2.0.0"
  pip install "mmcv-full"
3. Instalar MMDetection:
  pip install "mmdet<3.0.0"
4. Instalar MMPose: \
  cd mmtracking \
  pip install -r requirements/build.txt \
  pip install -v -e . \

Hay una incompatibilidad entre el modelo de detección y el modelo de seguimiento, que queda pendiente de solucionar.

## Citations
MMPose:
```bibtex
@misc{mmpose2020,
    title={OpenMMLab Pose Estimation Toolbox and Benchmark},
    author={MMPose Contributors},
    howpublished = {\url{https://github.com/open-mmlab/mmpose}},
    year={2020}
}
```
MMTracking:
```bibtex
@misc{mmtrack2020,
    title={{MMTracking: OpenMMLab} video perception toolbox and benchmark},
    author={MMTracking Contributors},
    howpublished = {\url{https://github.com/open-mmlab/mmtracking}},
    year={2020}
}
```
