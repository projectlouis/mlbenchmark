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

    fc1  = mx.symbol.FullyConnected(data = flat, name='fc1', num_hidden=25)
    act1 = mx.symbol.Activation(data = fc1, name='relu1', act_type="relu")
    fc2  = mx.symbol.FullyConnected(data = act1, name='fc2', num_hidden=5)
    act2 = mx.symbol.Activation(data = fc2, name='relu2', act_type="relu")
    fc3  = mx.symbol.FullyConnected(data = act2, name='fc3', num_hidden=1)
    net  = mx.symbol.LinearRegressionOutput(data=fc3, label=outLabl, name='linreg1')
	
	
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

LEARNING_RATE = 0.001
BATCH_SIZE = 128;
path = "./data/"

filename_read = os.path.join(path,"airfoil_self_noise.csv")
df = pd.read_csv(filename_read,skiprows=1,na_values=['NA','?'])

encode_numeric_zscore(df,'Freq_Hz')
encode_numeric_zscore(df,'AoA_Deg')
encode_numeric_zscore(df,'Chord_m')
encode_numeric_zscore(df,'V_inf_mps')
encode_numeric_zscore(df,'displ_thick_m')

x_out,y_out = to_xy(df,'sound_db')

X_train, X_test, y_train, y_test = train_test_split(
    x_out, y_out, test_size=0.20, random_state=42)

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
		  num_epoch = 5000,
		  optimizer_params={'learning_rate':LEARNING_RATE},
		  batch_end_callback=mx.callback.Speedometer(1,50), 
		  epoch_end_callback=None, 
		  eval_metric='mse',
		  optimizer='adam')

		  
# Create a ND Iter Format
predIter = mx.io.NDArrayIter(data = [x_out], batch_size = BATCH_SIZE)

pred = model.predict(predIter);

predDF = pd.DataFrame(pred.asnumpy())
df2 = pd.concat([df,predDF,pd.DataFrame(y_out)],axis=1)

df2.columns = list(df.columns)+['pred','ideal']
print(df2)


# Create a Pandas Excel writer using XlsxWriter as the engine.
writer = pd.ExcelWriter('airfoil_regression_mxnet.xlsx', engine='xlsxwriter')

# Convert the dataframe to an XlsxWriter Excel object.
df2.to_excel(writer, sheet_name='Sheet1')

# Close the Pandas Excel writer and output the Excel file.
writer.save()
