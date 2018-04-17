import tensorflow as tf
from tensorflow.python.client import timeline
#import matplotlib.pyplot as plt
import numpy as np


from TimeLiner import *
from model_zoo.model_zoo import *
from external_model import *

wafer = Wafer()
RESHAPE_LEN=100 #vgg:224   
OUTPUT_CLASS = 9 # number 1 to 10 data
TFRECORD_PATH='/home/franksai/ai/datasets/wafer/tfrecord/LSWMD_sub7000.tfrecords'

RECORD_LAP=50
epec=3
train_data_index=3000
test_data_offset=100
batch_size =16
meta_graph_path='./logs/train/meta/my-model2.meta'
checkpoint_path='./logs/train/meta/my-model2'
timeline_path ='./logs/train/timeline/wnet_sub7000_training.json'
tfevent_path = "./logs/train/tfboard/"

config = tf.ConfigProto()
#config.gpu_options.per_process_gpu_memory_fraction = 0.9
#config.gpu_options.allow_growth = True

print("Collecting training data...")
wafer = read_tfrecord(TFRECORD_PATH,wafer,RESHAPE_LEN,OUTPUT_CLASS)
print("Finish collecting training data")
print "Waiting for tenSorflow initial..."

def compute_accuracy(v_xs, v_ys):
    global prediction
    y_pre = sess.run(prediction, feed_dict={xs: v_xs, keep_prob: 1})
    v_ys =  np.vstack(v_ys)
    y_pre=np.vstack(np.vstack(np.vstack(y_pre))) #this line for vgg16

    correct_prediction = tf.equal(tf.argmax(y_pre,1), tf.argmax(v_ys,1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    #tf.summary.scalar('accuracy',accuracy)

    result = sess.run(accuracy, feed_dict={xs: v_xs, ys: v_ys, keep_prob: 1})
    return result

#====== network =======================================
# define placeholder for inputs to network
xs = tf.placeholder(tf.float32, [None, RESHAPE_LEN, RESHAPE_LEN],name='input_P')   # 28x28
ys = tf.placeholder(tf.float32, [None, OUTPUT_CLASS])
keep_prob = tf.placeholder(tf.float32)
x_image = tf.reshape( xs , [-1, RESHAPE_LEN, RESHAPE_LEN, 1])

netout = wnet( x_image,OUTPUT_CLASS)
#netout, end_points = inception.inception_v4( x_image,num_classes=OUTPUT_CLASS)
prediction = tf.nn.softmax(netout) #tensor name, Softmax:0

# the error between prediction and real data
prediction=tf.to_float(prediction, name='ToFloat')
tmp_Aa=tf.log( prediction)
loss_value = -tf.reduce_sum(ys * tmp_Aa,reduction_indices=[1] )
cross_entropy = tf.reduce_mean(loss_value ) #loss

#define optimizer
train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)


#tf initial
sess = tf.Session(config=config)
init = tf.global_variables_initializer()
sess.run(init)

#====for logs==============================================
saver = tf.train.Saver()
saver.save(sess, checkpoint_path)
run_metadata = tf.RunMetadata()
many_runs_timeline = TimeLiner()

#for tfboard
train_writer = tf.summary.FileWriter(tfevent_path, sess.graph) #for tf board
tf.summary.scalar('loss',cross_entropy) #record loss to tf board
merged = tf.summary.merge_all()

#==== Training ==============================================
for j in range(epec):
    for i in range(train_data_index):
        ran_index = i +randint(0,99)
        summary, _ = sess.run([merged,train_step],
                feed_dict = {xs: wafer.img[ran_index:ran_index+batch_size] , ys: wafer.label[ran_index:ran_index+batch_size], keep_prob: 0.5},
                options   = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE),
                run_metadata = run_metadata
                )
        #get timeline for each batch
        # #train_writer.add_run_metadata(run_metadata,'step%d' %(i+j*train_data_index))
        train_writer.add_summary(summary,(i+j*train_data_index))#record summary
        fetched_timeline = timeline.Timeline(run_metadata.step_stats)
        chrome_trace = fetched_timeline.generate_chrome_trace_format()
        many_runs_timeline.update_timeline(chrome_trace)

        if i % RECORD_LAP == 0:
            test_case =    wafer.img[train_data_index:train_data_index+test_data_offset]
            test_label = wafer.label[train_data_index:train_data_index+test_data_offset]

            print "epec: %d, step: %d,\taccuracy: %.5f"  %( j,i, compute_accuracy(test_case,test_label  ) )
            meta_graph_def = tf.train.export_meta_graph(filename=meta_graph_path) 
            many_runs_timeline.save(timeline_path )
            print "log done"

train_writer.close()
