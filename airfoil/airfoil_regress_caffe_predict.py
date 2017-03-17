import numpy as np
import sys
import caffe
import os
import pandas as pd

#http://www.programcreek.com/python/example/82811/caffe.TEST
#https://prateekvjoshi.com/2016/02/23/deep-learning-with-caffe-in-python-part-iv-classifying-an-image/

path = "./data/"
# Encode a numeric column as zscores
def encode_numeric_zscore(df,name,mean=None,sd=None):
    if mean is None:
        mean = df[name].mean()

    if sd is None:
        sd = df[name].std()

    df[name] = (df[name]-mean)/sd

def to_xy(df,target):
    result = []
    for x in df.columns:
        if x != target:
            result.append(x)

    # find out the type of the target column.  Is it really this hard? :(
    target_type = df[target].dtypes
    target_type = target_type[0] if hasattr(target_type, '__iter__') else target_type
    
    # Encode to int for classification, float otherwise. TensorFlow likes 32 bits.
    if target_type in (np.int64, np.int32):
        # Classification
        return df.as_matrix(result).astype(np.float32),df.as_matrix([target]).astype(np.int32)
    else:
        # Regression
        return df.as_matrix(result).astype(np.float64),df.as_matrix([target]).astype(np.float64)

	
pretrain = str("airfoil_iter_58710.caffemodel")
model_file = str("airfoil/mlp_auto_train.prototxt")

net = caffe.Net(str(model_file),str(pretrain),caffe.TEST)
#net.set_mode_cpu()


filename_read = os.path.join(path,"airfoil_self_noise.csv")
df = pd.read_csv(filename_read,skiprows=1,na_values=['NA','?'])

encode_numeric_zscore(df,'Freq_Hz')
encode_numeric_zscore(df,'AoA_Deg')
encode_numeric_zscore(df,'Chord_m')
encode_numeric_zscore(df,'V_inf_mps')
encode_numeric_zscore(df,'displ_thick_m')

x_out,y_out = to_xy(df,'sound_db')

pred = net.predict(x_out)

print(prediction);

predDF = pd.DataFrame(pred)
df2 = pd.concat([df,predDF,pd.DataFrame(y_out)],axis=1)

df2.columns = list(df.columns)+['pred','ideal']
print(df2)
