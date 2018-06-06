
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from Wafer import * 
from PIL import Image
import cv2

import operator
from random import randint as randint


#labelList=["Center" ,"Do", "Edge-Loc","Edge-Ring", "Loc", "Ran","Scratch","none","Near"]
labelList=['none', 'Loc', 'Edge-Loc', 'Center', 'Edge-Ring', 'Scratch', 'Random', 'Near-full', 'Donut']


#===========================================================================================

def print_foo():
    print "foo"



def print_allresize(raw_img):
    print "INTER_NEAREST"
    res = cv2.resize(raw_img, dsize=(500, 500), interpolation=cv2.INTER_NEAREST)
    plt.imshow( res )
    plt.show()
    print "INTER_LINEAR"
    res = cv2.resize(raw_img, dsize=(500, 500), interpolation=cv2.INTER_LINEAR)
    plt.imshow( res )
    plt.show()
    print "INTER_AREA"
    res = cv2.resize(raw_img, dsize=(500, 500), interpolation=cv2.INTER_AREA)
    plt.imshow( res )
    plt.show()
    print "INTER_CUBIC"
    res = cv2.resize(raw_img, dsize=(500, 500), interpolation=cv2.INTER_CUBIC)
    plt.imshow( res )
    plt.show()
    print "INTER_LANCZOS4"
    res = cv2.resize(raw_img, dsize=(500, 500), interpolation=cv2.INTER_LANCZOS4)
    plt.imshow( res )
    plt.show()

def read_tfrecord(TFRECORD_PATH,wafer,RESHAPE_LEN,OUTPUT_CLASS):
    label_1d=[0,0,0,0,0,0,0,0,0]
    record_iterator = tf.python_io.tf_record_iterator(path=TFRECORD_PATH)
    i=0
    for string_record in record_iterator:
        example = tf.train.Example()
        example.ParseFromString(string_record)
        #get image shape die_size
        x = int(example.features.feature['image_x'].int64_list.value[0])
        y = int(example.features.feature['image_y'].int64_list.value[0])
        d = int(example.features.feature['diesize'].int64_list.value[0])
        wafer.image_x.append( x)
        wafer.image_y.append( y)
        wafer.die_size.append(d)

        #get image
        image_string = (example.features.feature['img_raw'].bytes_list.value[0])
        image_1d = np.fromstring(image_string, dtype=np.uint8)
        # cv resize function
        image_2d = image_1d.reshape(x,y)
        image_2d_nromal = cv2.resize(image_2d, dsize=(RESHAPE_LEN, RESHAPE_LEN), interpolation=cv2.INTER_NEAREST)
        wafer.img.append( [image_2d_nromal] )

        #get lables
        tmp = int(example.features.feature['label'].int64_list.value[0])
        label_1d[tmp]=1
        wafer.label.append( np.array(label_1d).reshape((1,OUTPUT_CLASS))  )
        label_1d[tmp]=0
        i = i+1
        if i % 1000 == 0:
            print "data loading... index%d"% i 
    wafer.label = np.vstack(wafer.label)
    wafer.img   = np.vstack(wafer.img)
    return wafer
