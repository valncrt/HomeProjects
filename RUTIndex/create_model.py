#import keras
import pandas as pd
import math as math

#from index_data_prep import get_out_file

file ="C:\\Users\\csvai\\OneDrive\\Desktop\\Stephen\\RUTIndex\\data\\RUT_30_years_high_multiplier.csv"

pd=pd.read_csv(file)
#print(pd.High)

print(pd.size)
batch_size=13 #number of weeks to included to make prediction
size_for_even_batch_division=math.floor(pd.size/batch_size)*batch_size -(pd.size%batch_size)*batch_size #don't run over the end
print("size_for_even_batch_division",size_for_even_batch_division)


for i in range (0, size_for_even_batch_division, 1):
    print("I=",i,pd.High[i:i+batch_size])

#for x in range(0,1,pd.size ):
#    print("test x",x)