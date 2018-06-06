# Copyright 2018 franksai. All Rights Reserved.
# ==============================================================================
# review image in  dataset


import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from Wafer import *
from external_model import *

wafer = Wafer()
label_1d=[0,0,0,0,0,0,0,0,0]
OUTPUT_CLASS=9
labelList=["Center" ,"Do", "Edge-Loc","Edge-Ring", "Loc", "Ran","Scratch","none","Near"]

def next_batch(num, data, labels):
    '''
    Return a total of `num` random samples and labels. 
    '''
    idx = np.arange(0 , len(data))
    np.random.shuffle(idx)
    idx = idx[:num]
    data_shuffle = [data[i] for i in idx]
    labels_shuffle = [labels[ i] for i in idx]

    return np.asarray(data_shuffle), np.asarray(labels_shuffle) ,idx



record_iterator = tf.python_io.tf_record_iterator(path='./tfrecord/LSWMD_sub50.tfrecords')
for string_record in record_iterator:
    example = tf.train.Example()
    example.ParseFromString(string_record)

    #get image
    image_string = (example.features.feature['img_raw'].bytes_list.value[0])
    image_1d = np.fromstring(image_string, dtype=np.uint8)
    wafer.img.append( image_1d )#.reshape((1, 2160))) #45 48

    #get image shape die_size
    wafer.image_x.append(  int(example.features.feature['image_x'].int64_list.value[0]))
    wafer.image_y.append(  int(example.features.feature['image_y'].int64_list.value[0]))
    wafer.die_size.append( int(example.features.feature['diesize'].int64_list.value[0]))

    #get lables
    tmp = int(example.features.feature['label'].int64_list.value[0])
    label_1d[tmp]=1
    wafer.label.append( np.array(label_1d).reshape((1,OUTPUT_CLASS))  )
    label_1d[tmp]=0



ID =20

raw_img = wafer.img[ID].reshape(wafer.image_x[ID],wafer.image_y[ID])
plt.imshow( raw_img )
plt.show()
#print_allresize(raw_img) #print image
print("-----------------")
print("wafer label: ",wafer.label[ID]  )
print "wafer shape: " ,wafer.image_x[ID] , wafer.image_y[ID]
print "die size: " ,wafer.die_size[ID]
print("-----------------")
#plt.imshow( wafer.img[ID].reshape(wafer.image_x[ID] , wafer.image_y[ID]) )
#plt.show()
#labelList=["Center" ,"Do", "Edge-Loc","Edge-Ring", "Loc", "Ran","Scratch","none","Near"]


'''a part in dataset
(2160, 1683) 45 48
(3074, 2460)
(3068, 2393)
(1368, 1075) :2050
(1020, 776)
(1156, 904)
(900, 693)
(2496, 1999) :249
(3472, 2742) :68
(3135, 2491) :50
(6460, 5095) :3
(784, 607) :8
(2640, 2085) :135
(2688, 2072) :43
(7128, 5633) :20
(3135, 2513) :66
(7200, 5815) :94
(780, 600) :1513
(754, 562) :11252
(1804, 1376) :6134
(1722, 1334) :1460
(1848, 1414) :6723
...
'''
