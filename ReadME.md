This repo provide a wafer-inseption pototype include all tensorflow profiler tools.(tensorboard, timeline)
You can also use NVprofiler(.nvvp) to moniter your model performance without overhead.<br>

You can easiely import an other neural network to your application by modify `model_zoo`.
The coding style is friendly for beginer.


You can download dataset here: https://www.kaggle.com/qingyi/wm811k-wafer-map  <br>
In the link, is a `.pkl` file.
We provide a `pkl2tfrecord.py` script in `dataset`
After turn dataset to tfrecord, you can start your trraining.
