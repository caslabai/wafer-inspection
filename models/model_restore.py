import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import sys
from tensorflow.examples.tutorials.mnist import input_data
from tensorflow.python.client import timeline




mnist = input_data.read_data_sets('MNIST_data', one_hot=True)
#testdata.img  = mnist.test.images[1]  #print(sys.argv[1]) 
testdata ={ 'img':mnist.test.images , 'label':mnist.test.labels }
print(testdata['img'])
#plt.imshow( img.reshape([28, 28]) )
#plt.show()

with tf.Session() as sess:

    saver = tf.train.import_meta_graph('logs/my-model.meta')
    saver.restore(sess,tf.train.latest_checkpoint('./logs'))
    net = tf.get_default_graph()

    input_P  = net.get_tensor_by_name('input_P:0')
    output_D = net.get_tensor_by_name('Softmax:0')
    output = tf.argmax(output_D,axis=1)


    golden  = testdata['label'][1]
    run_metadata = tf.RunMetadata()

    predict = sess.run(output,feed_dict={input_P:testdata['img'][1].reshape([-1,784])}
                                        ,options=tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
                                        ,run_metadata=run_metadata)

    with  open('timeline_testing.ctf.json', 'w') as trace_file :
        trace = timeline.Timeline(step_stats=run_metadata.step_stats)
        trace_file.write(trace.generate_chrome_trace_format())


    print("golden ", np.argmax(golden) )
    print("predict ",predict[0])
    
    run_metadata = tf.RunMetadata()
    trace = timeline.Timeline(step_stats=run_metadata.step_stats)
    trace_file = open('./log/test/timeline.ctf.json', 'w')
    trace_file.write(trace.generate_chrome_trace_format())

 


