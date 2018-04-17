This repo provide a wafer-inseption pototype include all tensorflow profiler tools.(tensorboard, timeline)
You can also use NVprofiler(.nvvp) to moniter your model performance without overhead.<br>

You can easiely import an other neural network to your application by modify `model_zoo`.
The coding style is friendly for beginer.

# Get start

Few folder for log info. you need to build.<br>
`
mkdir logs/train/meta
mkdir logs/train/tfbord
mkdir logs/train/timeline
mkdir logs/test/tfbord
mkdir logs/test/timeline
`<br>

Execute `pkl2tfrecord.py` in `dataset` to build dataset, or use the sub dataset We provide in this repo<br>
Execute `model_train.py` to start training. But make sure your dataset is place on the right `TRRECORD_PATH`<br>
Execute `model_restore_wafer.py` can get the trained model to do the final test.<br>

# option
if you need full dataset:<br>
You can download dataset here: https://www.kaggle.com/qingyi/wm811k-wafer-map  <br>
In the link, is a `.pkl` file.
We provide a `pkl2tfrecord.py` script in `dataset`
After turn dataset into tfrecord, you can start your training.

# profiler
for tensorflow timeline <br>
use chrome for`chrome://tracing`, load file `logs/test/timeline` <br>

for tensorboard <br>
use cmd in `logs/test/tfbord`  <br>
`tensorboard --logdir=./` <br>
use browser goto `localhost:6006`<br> 

for nvprof (nvidia virtual profiler)<br>
`nvprof [-f] -o ./logs/nvvp/name.nvvp python model_train.py`<br>
<br>
