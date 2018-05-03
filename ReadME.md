This repo provide a wafer-inseption pototype include all tensorflow profiler tools.(tensorboard, timeline)<br>
PS. Loging hardware message by timline cause huge overhead, pls turn it off if you don't need it.<br>
You can also use NVprofiler(.nvvp) to moniter your model performance without overhead.<br>

You can easiely import an other neural network to your application by modify `model_zoo`.
The coding style is friendly for beginer.<br>

run under: python 2.7 in tensorflow1.4 cudnn6.1 cuda8.0<br>

# Get start

Few folder for log info. you need to build.<br>

`mkdir -p logs/train/meta`  <br>
`mkdir -p logs/train/tfbord` <br>
`mkdir -p logs/train/timeline` <br>
`mkdir -p logs/test/tfbord` <br>
`mkdir -p logs/test/timeline`<br>
<br>

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
## tensorflow timeline <br>
use chrome for`chrome://tracing`, load file `logs/test/timeline` <br>

## tensorboard <br>
use cmd in `logs/test/tfbord`  <br>
`tensorboard --logdir=./` <br>
use browser goto `localhost:6006`<br> 

## nvprof (nvidia virtual profiler)<br>
`nvprof [-f] -o ./logs/nvvp/name.nvvp python model_train.py`<br>
<br>

# Current result
 #accuracy up to 97% 
 
`
           70000 training data 10w testing data
           resnet 50 epec
                   train accuracy:  0.9894
                   test accuracy:   0.948283950617
  
      70000 training data 10w testing data
           resnet 50 epec
                   train accuracy:  0.9894
                   test accuracy:   0.948283950617
            wnet 82 epec
                   train accuracy:  0.9977
                   test accuracy:   0.973216049383
  
          inception_v4
                  train accuracy:  0.9937
                  test accuracy:   0.975475308642
`
