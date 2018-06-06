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

#%%
TEST_START =0#100000
TEST_BATCH=1000
train_index=10000


#%% to match training model
RESHAPE_LEN=150 #150 #vgg:224   
OUTPUT_CLASS = 9 # number 1 to 10 data
TMP=''#'TMP__'
model_name='inception/'  #'resnet/'  #'wnet/'#'vgg16'#'inception'
dataset = 'full'#'sub7000' #sub7000
RECORD_LAP=50
epec=100
train_data_index=70000
#test_data_offset=1000
batch_size =16


log_name=model_name +"newset2_sh_dropout_%dtraindata_%de_%dbatch_%dinputsize" %( train_data_index,epec,batch_size,RESHAPE_LEN )
meta_graph_path='./logs/train/meta/'+TMP+ log_name+'.meta'
checkpoint_path='./logs/train/meta/'+model_name
tfevent_path =  './logs/test/tfboard/'+TMP+log_name
timeline_path = './logs/test/timeline/wnet_'+dataset+'_training.json'
TFRECORD_PATH = '../dataset/tfrecord/new_LSWMD_'+dataset+'.tfrecords'
 
def read_data():
    wafer = Wafer()
    wafer = read_tfrecord(TFRECORD_PATH,wafer,RESHAPE_LEN,OUTPUT_CLASS)
    return wafer

def all_test(wafer,p_tag):
    config = tf.ConfigProto()
    #config.gpu_options.per_process_gpu_memory_fraction = 0.9
    config.gpu_options.allow_growth = True
   
    total_result=[]
    
    with tf.Session(config=config) as sess:
    
        saver = tf.train.import_meta_graph(meta_graph_path)
        saver.restore(sess,tf.train.latest_checkpoint(checkpoint_path))
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
            tmp =( np.array(result) == np.array(predict))
            total_result.append( tmp.mean() )
            if p_tag == 1:
		print "at step%d, total_accuracy: %.5f r_lenth: %d" %( i*TEST_BATCH ,np.array(total_result).mean() , len(total_result) )
            
            fetched_timeline = timeline.Timeline(run_metadata.step_stats)
            chrome_trace = fetched_timeline.generate_chrome_trace_format()
            many_runs_timeline.update_timeline(chrome_trace)
    
        tf.summary.FileWriter(tfevent_path, sess.graph)
        many_runs_timeline.save(timeline_path )
       
        train_ac = np.array(total_result[: (train_index/TEST_BATCH) ]).mean()
        test_ac = np.array(total_result[(train_index/TEST_BATCH) : nround]).mean()
    
        print "train accuracy: ",train_ac
        print "test accuracy:  ",test_ac	
     	return train_ac , test_ac

if __name__ == '__main__':
    datas = read_data()
    all_test(datas,0)
else:
    all_test()


