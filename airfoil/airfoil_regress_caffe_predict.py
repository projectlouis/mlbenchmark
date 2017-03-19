import numpy as np
import sys
import caffe
import os
import pandas as pd
import pdb;
#import barrista
#from barrista import design

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

#barristaNet = barrista.design.NetSpecification.from_prototxt(filename=pretrain)

#net = barristaNet.instantiate()

filename_read = os.path.join(path,"airfoil_self_noise.csv")
df = pd.read_csv(filename_read,skiprows=1,na_values=['NA','?'])

encode_numeric_zscore(df,'Freq_Hz')
encode_numeric_zscore(df,'AoA_Deg')
encode_numeric_zscore(df,'Chord_m')
encode_numeric_zscore(df,'V_inf_mps')
encode_numeric_zscore(df,'displ_thick_m')

x_out,y_out = to_xy(df,'sound_db')

#net.forward(x_out)
print(net)
print(net.blobs)
print(net.blobs['data'])
print(net.blobs['data'].data.shape)

for layer_name, blob in net.blobs.iteritems():
    print layer_name + '\t' + str(blob.data.shape)

# This is reshaping it...
#net.blobs['data'].reshape(*x_out.shape)


#print(net.blobs['data'].data.shape)
#for layer_name, blob in net.blobs.iteritems():
    #print layer_name + '\t' + str(blob.data.shape)

#print(x_out.shape)


batch_size = 128
pred = []

#for start, end in zip(range(0, len(x_out), batch_size), range(batch_size, len(x_out), batch_size)):
#start = 128
#end = 128+128
#net.blobs['data'].data[...] = x_out[start:end,:]
#output1 = net.forward(['fc3'])
#pred.append(output1['fc3'])
#print(output1)

for start, end in zip(range(0, len(x_out), batch_size), range(batch_size, len(x_out), batch_size)):
	net.blobs['data'].data[...] = x_out[start:end,:]
	output1 = net.forward(start='fc1',end='fc3')
	new_array = np.copy(output1['fc3'])
	pred.append(new_array)
	

nTotal = len(x_out);
start = end

newset = np.zeros((batch_size,5))
newset[0:nTotal-start,:] = x_out[start:nTotal+1,:]

net.blobs['data'].data[...] = newset
output1 = net.forward(start='fc1',end='fc3')
new_array = np.copy(output1['fc3'])
pred.append(new_array)



X = np.array(pred)

N,M,O = X.shape

pred = X.transpose(2,0,1).reshape(N*M,-1)

pred = pred[0:nTotal]

#pdb.set_trace()

print(pred);

predDF = pd.DataFrame(pred)
df2 = pd.concat([df,predDF,pd.DataFrame(y_out)],axis=1)

df2.columns = list(df.columns)+['pred','ideal']
print(df2)

# Create a Pandas Excel writer using XlsxWriter as the engine.
writer = pd.ExcelWriter('airfoil_regression_caffe.xlsx', engine='xlsxwriter')

# Convert the dataframe to an XlsxWriter Excel object.
df2.to_excel(writer, sheet_name='Sheet1')

# Close the Pandas Excel writer and output the Excel file.
writer.save()
