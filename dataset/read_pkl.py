#import pickle
import pandas as pd
import time

local_time = time.strftime("%Y-%m-%d_%H:%M:%S", time.localtime())
FILENAME="sub100_WM-811K_"+local_time
pkl_file = pd.read_pickle('./LSWMD.pkl')

sub_pkl =pkl_file.iloc[0:100,0:7]
print(sub_pkl )



print(pkl_file.ndim)
print("---") 
print(pkl_file.shape)
print("---") 
print(pkl_file.dtypes)

sub_pkl.to_csv(FILENAME, sep='\t', encoding='utf-8')

'''
build_file = open(FILENAME,"w")
build_file.write(sub_pkl)
build_file.close()
#print(aa.type())
'''

