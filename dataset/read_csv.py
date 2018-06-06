import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sys

def str2array(img ):
    img=img.replace(',', '' )
    img=img.replace(']', '' )
    img=img.replace('[', '' )
    img=img.replace('\n', '' )
    #img=img.replace('0', '' )
    #img=img.replace(' ', '' )

    print(img)
    print( "in str2array: ",len(img))
    img = img.split(' ')
    img =np.array([int(i) for i in img])
    #img = np.array(map(int,img) )
    return img

dieID =35
dataset = pd.read_csv('sub100_WM-811K.csv', sep='\t')
img = dataset.iloc[dieID]["waferMap"]#.values
failureType = dataset.iloc[dieID]["failureType"]#.values
img_size = dataset.iloc[dieID]["dieSize"]


#dieSize_list = dataset["dieSize"]
#print(dieSize_list)
#print(img) #type as str

img = str2array(img)
#print(img.aaa)
print(img_size)
print("failureType: ",failureType)

plt.imshow( img.reshape([53,58])) #([45, 48]) )
plt.show()
