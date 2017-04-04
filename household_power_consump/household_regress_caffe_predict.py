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

	
pretrain = str("household1_iter_320200.caffemodel")
model_file = str("household/mlp_auto_train.prototxt")

net = caffe.Net(str(model_file),str(pretrain),caffe.TEST)

### BELOW HERE CHANGED
LEARNING_RATE = 0.000001
BATCH_SIZE = 512
path = "./data/"

filename_read = os.path.join(path,"household_power_consump.csv")
df = pd.read_csv(filename_read,skiprows=1,na_values=['NA','?'])

encode_numeric_zscore(df,'Global_active_power')
encode_numeric_zscore(df,'Global_reactive_power')
encode_numeric_zscore(df,'Global_intensity')
#encode_numeric_zscore(df,'Voltage')
encode_numeric_zscore(df,'Sub_metering_1')
encode_numeric_zscore(df,'Sub_metering_2')
encode_numeric_zscore(df,'Sub_metering_3')

del df['Date']
del df['Time']

print(df)

x_out,y_out = to_xy(df,'Voltage')
#x_out,y_out = to_xy(df,'GT_compre_coef')
## ABOVE HERE CHANGED


#net.forward(x_out)
print(net)
print(net.blobs)
print(net.blobs['data'])
print(net.blobs['data'].data.shape)

for layer_name, blob in net.blobs.iteritems():
    print layer_name + '\t' + str(blob.data.shape)

batch_size = 512

numPartition = 8

new_xout = np.array_split(x_out,numPartition)
new_yout = np.array_split(y_out,numPartition)

new_df = np.array_split(df,numPartition)

for ji in range(0,numPartition):
	print('Starting to predict')
	
	pred = []

	for start, end in zip(range(0, len(new_xout[ji]), batch_size), range(batch_size, len(new_xout[ji]), batch_size)):
		net.blobs['data'].data[...] = new_xout[ji][start:end,:]
		output1 = net.forward(start='fc1',end='fc5')
		new_array = np.copy(output1['fc5'])
		pred.append(new_array)

	nTotal = len(new_xout[ji]);
	start = end

	newset = np.zeros((batch_size,6))
	newset[0:nTotal-start,:] = new_xout[ji][start:nTotal+1,:]

	net.blobs['data'].data[...] = newset
	output1 = net.forward(start='fc1',end='fc5')
	new_array = np.copy(output1['fc5'])
	pred.append(new_array)

	X = np.array(pred)
	N,M,O = X.shape
	pred = X.transpose(2,0,1).reshape(N*M,-1)
	pred = pred[0:nTotal]
	###
	
	#pred = list(network.predict(new_xout[ji]));
    
	predDF = pd.DataFrame(pred)
	df2 = pd.concat([new_df[ji],predDF,pd.DataFrame(new_yout[ji])],axis=1)

	print('Starting to write to csv file')
	df2.to_csv("household_regression_caffe" + str(ji) + ".csv", chunksize=1000)



#pdb.set_trace()

#print(pred);

#predDF = pd.DataFrame(pred)
#df2 = pd.concat([df,predDF,pd.DataFrame(y_out)],axis=1)

#df2.columns = list(df.columns)+['pred','ideal']
#print(df2)

# Create a Pandas Excel writer using XlsxWriter as the engine.
#writer = pd.ExcelWriter('naval_regression_caffe_turbine.xlsx', engine='xlsxwriter')

# Convert the dataframe to an XlsxWriter Excel object.
#df2.to_excel(writer, sheet_name='Sheet1')

# Close the Pandas Excel writer and output the Excel file.
#writer.save()