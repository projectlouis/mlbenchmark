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
	fc1  = mx.symbol.FullyConnected(data = flat, name='fc1', num_hidden=500)
	act1 = mx.symbol.Activation(data = fc1, name='relu1', act_type="relu")
	fc2  = mx.symbol.FullyConnected(data = act1, name='fc2', num_hidden=250)
	act2 = mx.symbol.Activation(data = fc2, name='relu2', act_type="relu")
	fc3  = mx.symbol.FullyConnected(data = act2, name='fc3', num_hidden=5)
	act3 = mx.symbol.Activation(data = fc3, name='relu3', act_type="relu")
	fc4  = mx.symbol.FullyConnected(data = act3, name='fc4', num_hidden=1)
	net  = mx.symbol.LinearRegressionOutput(data=fc4, label=outLabl, name='linreg1')
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
BATCH_SIZE = 128;
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

df = df.drop('gt_in_airT',1)

x_out,y_out = to_xy(df,'GT_turb_coef')
#x_out,y_out = to_xy(df,'GT_compre_coef')

X_train, X_test, y_train, y_test = train_test_split(
    x_out, y_out, test_size=0.20, random_state=42)

X_train = X_train[:,0:15]
X_test = X_test[:,0:15]
x_out = x_out[:,0:15]
	
print(x_out.shape);
print(X_train.shape);
print(y_train.shape);
	
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
		  num_epoch = 1235,
		  optimizer_params={'learning_rate':LEARNING_RATE},
		  epoch_end_callback=None, 
		  eval_metric='mse',
		  optimizer='adam')

		  
# Create a ND Iter Format
predIter = mx.io.NDArrayIter(data = x_out, batch_size = BATCH_SIZE)

pred = model.predict(predIter);

predDF = pd.DataFrame(pred.asnumpy())
df2 = pd.concat([df,predDF,pd.DataFrame(y_out)],axis=1)

df2.columns = list(df.columns)+['pred','ideal']
print(df2)


# Create a Pandas Excel writer using XlsxWriter as the engine.
writer = pd.ExcelWriter('naval_regression_mxnet.xlsx', engine='xlsxwriter')

# Convert the dataframe to an XlsxWriter Excel object.
df2.to_excel(writer, sheet_name='Sheet1')

# Close the Pandas Excel writer and output the Excel file.
writer.save()