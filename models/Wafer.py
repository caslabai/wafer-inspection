from random import shuffle

class Wafer:

    def __init__(self):
        self.image_x =[]
        self.image_y =[]
        self.img  =[]
        self.label =[]
        self.die_size=[]

    	self.sh_img=[]
    	self.sh_label=[]

    def w_print(self):
        print "foo"

    def shuffle(self,data_range):
	print "shuffling..."		
	index_shuf = range(len(self.img[:data_range]))
	shuffle(index_shuf)
	for i in index_shuf:
	    self.sh_img.append(self.img[i])
	    self.sh_label.append(self.label[i])



