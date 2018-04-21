import tensorflow as tf
from tensorflow.python.client import timeline
#import matplotlib.pyplot as plt
import numpy as np
import re


from TimeLiner import *
from model_zoo.model_zoo import *
from external_model import *

wafer = Wafer()
RESHAPE_LEN=224 #150 #vgg:224   
OUTPUT_CLASS = 9 # number 1 to 10 data
TMP=''#'/TMP/'
model_name='wnet/' #'wnet/' 'vgg16' 'inception'
dataset = 'FULL'#'sub7000' #sub7000
RECORD_LAP=50
epec=100
train_data_index=10000
test_data_offset=1000
batch_size =16

log_name = model_name + "dropout_%dtraindata_%de_%dbatch_%dinputsize" %( train_data_index,epec,batch_size,RESHAPE_LEN )
#meta_graph_path='./logs/train/meta/'+TMP+ log_name+'.meta'
checkpoint_path='./logs/train/meta/'+TMP+ log_name
tfevent_path =  './logs/train/tfboard/'+TMP+log_name
timeline_path = './logs/train/timeline/wnet_'+dataset+'_training.json'
TFRECORD_PATH = '../dataset/tfrecord/LSWMD_'+dataset+'.tfrecords'

config = tf.ConfigProto()
#config.gpu_options.per_process_gpu_memory_fraction = 0.9
#config.gpu_options.allow_growth = True

print("Collecting training data...")
wafer = read_tfrecord(TFRECORD_PATH,wafer,RESHAPE_LEN,OUTPUT_CLASS)
print("Finish collecting training data")
wafer.shuffle(train_data_index)

print "Waiting for tenSorflow initial..."

def compute_accuracy(v_xs, v_ys,step):
    #global prediction
    #global merged
    y_pre = sess.run(prediction, feed_dict={xs: v_xs, keep_prob: 1})
    v_ys =  np.vstack(v_ys)
    y_pre=np.vstack(np.vstack(np.vstack(y_pre))) #this line for vgg16
    
    correct_prediction = tf.equal(tf.argmax(y_pre,1), tf.argmax(v_ys,1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    
    ac_summary ,result = sess.run([merged,accuracy], feed_dict={xs: v_xs, ys: v_ys, keep_prob: 1})
    #train_writer.add_summary(ac_summary,step)#record summary
    
    return ac_summary, result
    
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
#tmp_Aa=tf.log( prediction)
#loss_value = -tf.reduce_sum(ys * tmp_Aa,reduction_indices=[1] )
#cross_entropy = tf.reduce_mean(loss_value ) #loss
cross_entropy = -tf.reduce_sum(ys*tf.log(tf.clip_by_value(prediction,1e-10,1.0)))


#define optimizer
train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)

#tf initial
sess = tf.Session(config=config)
init = tf.global_variables_initializer()
sess.run(init)


#====for logs==============================================
saver = tf.train.Saver()
#saver.save(sess, checkpoint_path)
run_metadata = tf.RunMetadata()
many_runs_timeline = TimeLiner()

#for tfboard
accuracy=0
train_writer = tf.summary.FileWriter(tfevent_path, sess.graph) #for tf board
tf.summary.scalar('loss',cross_entropy) #record loss to tf board
#tf.summary.scalar('test_accuracy',accu)
#tf.summary.scalar('train_accuracy',) #record loss to tf board

merged = tf.summary.merge_all()

#==== Training ==============================================
step = train_data_index / batch_size

for j in range(epec):
    for i in range(step):
	ran_index = i*batch_size#  + randint(0,50)
	summary, _ = sess.run([merged,train_step],
		feed_dict = {xs: wafer.sh_img[ran_index:ran_index+batch_size] , ys: wafer.sh_label[ran_index:ran_index+batch_size], keep_prob: 0.5},
		options   = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE),
		run_metadata = run_metadata
		)
	#print cross_entropy 
	#get timeline for each batch
	train_writer.add_run_metadata(run_metadata,'step%d' %(i+j*step))
	train_writer.add_summary(summary,(i+j*step ))#record summary
	#fetched_timeline = timeline.Timeline(run_metadata.step_stats)
	#chrome_trace = fetched_timeline.generate_chrome_trace_format()
	#many_runs_timeline.update_timeline(chrome_trace)

	if i % RECORD_LAP == 0:
	    test_case =    wafer.img[train_data_index:train_data_index+test_data_offset]
	    test_label = wafer.label[train_data_index:train_data_index+test_data_offset]

	    _, accu = compute_accuracy(test_case,test_label,i )
	    ac_sum = tf.Summary(value=[
	    	tf.Summary.Value(tag="test accuracy", simple_value=accu),
    		#tf.Summary.Value(tag="summary_tag2", simple_value=1),
	    ])

	    train_writer.add_summary(ac_sum,(i+j*step ))#record summary
	    print "epec: %d, step: %d,\taccuracy: %.5f"  %( j,i+j*step,accu  )
	    #tf.summary.scalar('accuracy',accu) #record loss to tf board
	    
	    saver.save(sess, checkpoint_path)
	    #meta_graph_def = tf.train.export_meta_graph(filename=meta_graph_path) 
	    #many_runs_timeline.save(timeline_path )
	    print "log done"

train_writer.close()


#====Multi GPU======================================================
'''
def average_gradients(tower_grads):
  """Calculate the average gradient for each shared variable across all towers.
  Note that this function provides a synchronization point across all towers.
  Args:
    tower_grads: List of lists of (gradient, variable) tuples. The outer list
      is over individual gradients. The inner list is over the gradient
      calculation for each tower.
  Returns:
     List of pairs of (gradient, variable) where the gradient has been averaged
     across all towers.
  """
  average_grads = []
  for grad_and_vars in zip(*tower_grads):
    # Note that each grad_and_vars looks like the following:
    #   ((grad0_gpu0, var0_gpu0), ... , (grad0_gpuN, var0_gpuN))
    grads = []
    for g, _ in grad_and_vars:
      #print g
      if g is None :
	print "==========GGGGGG"
	return 0
	#continue
      # Add 0 dimension to the gradients to represent the tower.
      expanded_g = tf.expand_dims(g, 0)
      #print expanded_g 
      # Append on a 'tower' dimension which we will average over below.
      grads.append(expanded_g)

    #print">>>>>>>", grads
    # Average over the 'tower' dimension.
    grad = tf.concat(axis=0, values=grads)
    grad = tf.reduce_mean(grad, 0)

    # Keep in mind that the Variables are redundant because they are shared
    # across towers. So .. we will just return the first tower's pointer to
    # the Variable.
    v = grad_and_vars[0][1]
    grad_and_var = (grad, v)
    average_grads.append(grad_and_var)
  return average_grads

'''


'''
global_step = tf.get_variable(
        'global_step', [],
        initializer=tf.constant_initializer(0))

tower_grads = []

for k in range(3):
    with tf.device('/gpu:%d' % k):
	with tf.name_scope('%s_%d' % ("GPU_", k)) as scope:
		#netout = wnet( x_image,OUTPUT_CLASS)
		netout, end_points = inception.inception_v4( x_image,num_classes=OUTPUT_CLASS)
		prediction = tf.nn.softmax(netout) #tensor name, Softmax:0
		# Calculate the total loss for the current tower.
		prediction=tf.to_float(prediction, name='ToFloat')
		loss_value = -tf.reduce_sum(ys * tf.log( prediction),reduction_indices=[1] )
		cross_entropy = tf.reduce_mean(loss_value ) #loss
		#total_loss = losses
  		
		# Calculate the average cross entropy loss across the batch.
  		#labels = tf.cast(labels, tf.int64)
  		#cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(
      		#labels=labels, logits=logits, name='cross_entropy_per_example')
  		
		cross_entropy_mean = tf.reduce_mean(cross_entropy, name='cross_entropy')
		tf.add_to_collection('losses', cross_entropy_mean)


		losses = tf.get_collection('losses', scope)
		total_loss = tf.add_n(losses, name='total_loss')

		# Attach a scalar summary to all individual losses and the total loss; do the
		# same for the averaged version of the losses.
		for l in losses + [total_loss]:
			# Remove 'tower_[0-9]/' from the name in case this is a multi-GPU training
			# session. This helps the clarity of presentation on tensorboard.
			loss_name = re.sub('%s_[0-9]*/' % "GPU_", '', l.op.name)
			tf.summary.scalar(loss_name, l)


		# Reuse variables for the next tower.
		tf.get_variable_scope().reuse_variables()

		# Retain the summaries from the final tower.
		summaries = tf.get_collection(tf.GraphKeys.SUMMARIES, scope)

		# Calculate the gradients for the batch of data on this CIFAR tower.
		opt = tf.train.GradientDescentOptimizer(0.0001)
		grads = opt.compute_gradients(total_loss)
		if grads is None:
			print "error append"
		else :
			tower_grads.append(grads)
#print grads
grads = average_gradients(tower_grads)
# Add a summary to track the learning rate.
#summaries.append(tf.summary.scalar('learning_rate', lr))
# Apply the gradients to adjust the shared variables.
apply_gradient_op = opt.apply_gradients(grads, global_step=global_step)
# Group all updates to into a single train op.
train_op = tf.group(apply_gradient_op, variables_averages_op)
saver = tf.train.Saver(tf.global_variables())

# Build the summary operation from the last tower summaries.
summary_op = tf.summary.merge(summaries)

'''



