import mxnet as mx
import numpy as np
import pandas as pd
import os
import shutil
from sklearn.model_selection import train_test_split
import time
import logging

def build_mlp():
	"""
	multi-layer perceptron
	"""

	outLabl = mx.sym.Variable('softmax_label')
	data = mx.symbol.Variable('data')
	flat = mx.symbol.Flatten(data=data)
	fc1  = mx.symbol.FullyConnected(data = flat, name='fc1', num_hidden=64)
	act1 = mx.symbol.Activation(data = fc1, name='relu1', act_type="relu")
	fc2  = mx.symbol.FullyConnected(data = act1, name='fc2', num_hidden=512)
	act2 = mx.symbol.Activation(data = fc2, name='relu2', act_type="relu")
	fc3  = mx.symbol.FullyConnected(data = act2, name='fc3', num_hidden=1024)
	act3 = mx.symbol.Activation(data = fc3, name='relu3', act_type="relu")
	fc4  = mx.symbol.FullyConnected(data = act3, name='fc4', num_hidden=128)
	act4 = mx.symbol.Activation(data = fc4, name='relu4', act_type="relu")
	fc5  = mx.symbol.FullyConnected(data = act4, name='fc5', num_hidden=1)
	net  = mx.symbol.LinearRegressionOutput(data=fc5, label=outLabl, name='linreg1')
	return net

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

def get_model_dir(name,erase):
    base_path = os.path.join(".","saved_models")
    model_dir = os.path.join(base_path,name)
    os.makedirs(model_dir,exist_ok=True)
    if erase and len(model_dir)>4 and os.path.isdir(model_dir):
        shutil.rmtree(model_dir,ignore_errors=True) # be careful, this deletes everything below the specified path
    return model_dir

def iterate_minibatches(inputs, targets, batchsize, shuffle=True):
    assert len(inputs) == len(targets)
    if shuffle:
        indices = np.arange(len(inputs))
        np.random.shuffle(indices)
    for start_idx in range(0, len(inputs) - batchsize + 1, batchsize):
        if shuffle:
            excerpt = indices[start_idx:start_idx + batchsize]
        else:
            excerpt = slice(start_idx, start_idx + batchsize)
        yield inputs[excerpt], targets[excerpt]
	
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

#print(df)

x_out,y_out = to_xy(df,'Voltage')
## ABOVE HERE CHANGED
#x_out,y_out = to_xy(df,['Sub_metering_1','Sub_metering_2','Sub_metering_3'])

X_train, X_test, y_train, y_test = train_test_split(
    x_out, y_out, test_size=0.20, random_state=42)

#print(df)
print(y_train.shape)
print(X_train.shape)
	
	
trainIter = mx.io.NDArrayIter(data = X_train, label = y_train, batch_size = BATCH_SIZE)
valIter   = mx.io.NDArrayIter(data = X_test , label = y_test , batch_size = BATCH_SIZE)


print("Building model and compiling functions...")
#
# Get model and train
#
net = build_mlp()

model = mx.mod.Module(symbol = net,
		label_names = ['softmax_label'],
		data_names = ['data'])

logging.basicConfig(level=logging.INFO)

print("Fitting Network...")
model.fit(trainIter, 
		  eval_data=valIter,
		  num_epoch = 100,
		  optimizer_params={'learning_rate':LEARNING_RATE},
		  epoch_end_callback=None, 
		  eval_metric='mse',
		  optimizer='adam')

numPartition = 8

new_xout = np.array_split(x_out,numPartition)
new_yout = np.array_split(y_out,numPartition)

new_df = np.array_split(df,numPartition)

for ji in range(0,numPartition):
	print('Starting to predict')
	predIter = mx.io.NDArrayIter(data = new_xout[ji], batch_size = BATCH_SIZE)
	pred = model.predict(predIter);
    
	predDF = pd.DataFrame(pred.asnumpy())
	df2 = pd.concat([new_df[ji],predDF,pd.DataFrame(new_yout[ji])],axis=1)

	print('Starting to write to csv file')
	df2.to_csv("household_regression_mxnet" + str(ji) + ".csv", chunksize=1000)


