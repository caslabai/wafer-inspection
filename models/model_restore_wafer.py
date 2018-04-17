# Copyright 2018 franksai. All Rights Reserved.
# ==============================================================================
# restore trained model to do inference
# go through all testing or training data to see accuracy
 
#%%
''' #print current dictionary 
import os
os.getcwd()
os.chdir("/home/franksai/ai/app/img_classify")
os.getcwd()
'''
#%%
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import sys
from tensorflow.examples.tutorials.mnist import input_data
from tensorflow.python.client import timeline
from Wafer import *
from external_model import *
import operator
from TimeLiner import *


config = tf.ConfigProto()
#config.gpu_options.per_process_gpu_memory_fraction = 0.9
config.gpu_options.allow_growth = True

#%%
wafer = Wafer()
RESHAPE_LEN=100
OUTPUT_CLASS = 9  # number 1 to 10 data
timeline_path ='./logs/test/timeline/_test_timeline.ctf.json'
meta_path ="./logs/train/meta/my-model2.meta"
checkp_path='./logs/train/meta'
tfevent_path = "./logs/test/tfboard/"

#%% read dataset
TFRECORD_PATH='/home/franksai/ai/datasets/wafer/tfrecord/LSWMD_sub7000.tfrecords'
wafer = read_tfrecord(TFRECORD_PATH,wafer,RESHAPE_LEN,OUTPUT_CLASS)
#%%
TEST_START =0#100000
TEST_BATCH=2000

total_result=[]

with tf.Session(config=config) as sess:

    saver = tf.train.import_meta_graph(meta_path)
    saver.restore(sess,tf.train.latest_checkpoint(checkp_path))
    net = tf.get_default_graph()

    input_P  = net.get_tensor_by_name('input_P:0')
    output_D = net.get_tensor_by_name('Softmax:0')
    output = tf.argmax(output_D,axis=1)

    run_metadata = tf.RunMetadata()
    many_runs_timeline = TimeLiner()

    nround =len(wafer.img)/TEST_BATCH
    for i in range(0, nround ):
        TEST_START=i*TEST_BATCH
        predict = sess.run(output,  feed_dict={input_P: wafer.img[TEST_START:TEST_START+ TEST_BATCH] },
                                    options   = tf.RunOptions(
                                        trace_level=tf.RunOptions.FULL_TRACE),
                                    run_metadata = run_metadata
                            )
        # trace_level = FULL_TRACE,HARDWARE_TRACE,NO_TRACE,SOFTWARE_TRACE
        
        result=[]
        for max_arg in wafer.label[TEST_START:TEST_START+TEST_BATCH]:
            result.append(max_arg.argmax())
        tmp = np.array(result) == np.array(predict)
        total_result.append( tmp.mean() )
        print "at step%d, total_accuracy: %.5f r_lenth: %d" %( i*2000,np.array(total_result).mean() , len(total_result) )
        
        fetched_timeline = timeline.Timeline(run_metadata.step_stats)
        chrome_trace = fetched_timeline.generate_chrome_trace_format()
        many_runs_timeline.update_timeline(chrome_trace)

    tf.summary.FileWriter(tfevent_path, sess.graph)
    many_runs_timeline.save(timeline_path )
    
    print "train accuracy: ",np.array(total_result[:2]).mean()
    print "test accuracy:  ",np.array(total_result[2:nround]).mean()





