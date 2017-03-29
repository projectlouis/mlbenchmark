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

	
pretrain = str("naval2_iter_82700.caffemodel")
model_file = str("naval/mlp_auto_train.prototxt")

net = caffe.Net(str(model_file),str(pretrain),caffe.TEST)

### BELOW HERE CHANGED
LEARNING_RATE = 0.000001
BATCH_SIZE = 128
path = "./data/"

filename_read = os.path.join(path,"naval_propulsion.csv")
df = pd.read_csv(filename_read,skiprows=1,na_values=['NA','?'])

encode_numeric_zscore(df,'Lever_position')
encode_numeric_zscore(df,'ship_speed')
encode_numeric_zscore(df,'gt_torque')
encode_numeric_zscore(df,'gt_revs')
encode_numeric_zscore(df,'gg_revs')
encode_numeric_zscore(df,'star_prop_torque')
encode_numeric_zscore(df,'port_prop_torque')
encode_numeric_zscore(df,'hp_exit_temp')
encode_numeric_zscore(df,'gt_out_airT')
encode_numeric_zscore(df,'HP_exit_pres')
encode_numeric_zscore(df,'GT_in_pres')
encode_numeric_zscore(df,'GT_out_pres')
encode_numeric_zscore(df,'gas_exhaust_pres')
encode_numeric_zscore(df,'turb_injec')
encode_numeric_zscore(df,'fuel_flow')

del df['gt_in_airT']

print(df)

x_out,y_out = x_out,y_out = to_xy(df,['GT_compre_coef','GT_turb_coef'])
x_out = x_out[:,0:15]


#net.forward(x_out)
print(net)
print(net.blobs)
print(net.blobs['data'])
print(net.blobs['data'].data.shape)

for layer_name, blob in net.blobs.iteritems():
    print layer_name + '\t' + str(blob.data.shape)

batch_size = 128
pred = []

for start, end in zip(range(0, len(x_out), batch_size), range(batch_size, len(x_out), batch_size)):
	net.blobs['data'].data[...] = x_out[start:end,:]
	output1 = net.forward(start='fc1',end='fc4')
	new_array = np.copy(output1['fc4'])
	pred.append(new_array)
	

nTotal = len(x_out);
start = end

newset = np.zeros((batch_size,15))
newset[0:nTotal-start,:] = x_out[start:nTotal+1,:]

net.blobs['data'].data[...] = newset
output1 = net.forward(start='fc1',end='fc4')
new_array = np.copy(output1['fc4'])
pred.append(new_array)


X = np.array(pred)

N,M,O = X.shape

pred = X.transpose(2,0,1).reshape(N*M,-1)

pred = pred[0:nTotal]

#pdb.set_trace()

print(pred);

predDF = pd.DataFrame(pred)
df2 = pd.concat([df,predDF,pd.DataFrame(y_out)],axis=1)

#df2.columns = list(df.columns)+['pred','ideal']
#print(df2)

# Create a Pandas Excel writer using XlsxWriter as the engine.
writer = pd.ExcelWriter('naval_regression_caffe.xlsx', engine='xlsxwriter')

# Convert the dataframe to an XlsxWriter Excel object.
df2.to_excel(writer, sheet_name='Sheet1')

# Close the Pandas Excel writer and output the Excel file.
writer.save()