import theano
from theano import tensor as T
from theano.sandbox.rng_mrg import MRG_RandomStreams as RandomStreams
import numpy as np
import pandas as pd
import os
import shutil
from sklearn.model_selection import train_test_split
import lasagne
from lasagne import layers
from nolearn.lasagne import NeuralNet
from nolearn.lasagne import BatchIterator
from nolearn.lasagne import TrainSplit
from lasagne import nonlinearities
import time

srng = RandomStreams()


def build_mlp(input_var=None):
## The input # needs to change according to # of inputs
    l_in = lasagne.layers.InputLayer(shape=(None, 15), input_var=input_var)
    l_hid1 = lasagne.layers.DenseLayer(
            l_in, num_units=500,
            nonlinearity=nonlinearities.rectify)
    l_hid2 = lasagne.layers.DenseLayer(
            l_hid1, num_units=250,
            nonlinearity=nonlinearities.rectify)
    l_hid3 = lasagne.layers.DenseLayer(
            l_hid2, num_units=5,
            nonlinearity=nonlinearities.rectify)
    l_out = lasagne.layers.DenseLayer(
            l_hid3, num_units=2,
            nonlinearity=None)
			
    net = NeuralNet(l_out, 
					regression=True,
					update_learning_rate = 0.000001,
					batch_iterator_train = BatchIterator(batch_size=128),
					batch_iterator_test = BatchIterator(batch_size=128),
					update=lasagne.updates.adam,
					max_epochs = 1235,
					train_split = TrainSplit(eval_size=0.2),
					objective_loss_function = lasagne.objectives.squared_error,
					verbose=1)
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

#x_out,y_out = to_xy(df,'GT_turb_coef')
x_out,y_out = to_xy(df,['GT_compre_coef','GT_turb_coef'])

## ABOVE HERE CHANGED
X_train, x_test, y_train, y_test = train_test_split(
    x_out, y_out, test_size=0, random_state=42)

print(x_out.shape);
print(X_train.shape);
print(y_train.shape);
	
X_train = X_train[:,0:15]
x_test = x_test[:,0:15]
x_out = x_out[:,0:15]

print(x_out.shape);
print(X_train.shape);
print(y_train.shape);

X = T.matrix()
Y = T.matrix()
	
print("Building model and compiling functions...")
network = build_mlp(X)

print("Fitting Network...")
network.fit(X_train,y_train);

print(network.score(X_train,y_train));


pred = list(network.predict(x_out));
predDF = pd.DataFrame(pred)
df2 = pd.concat([df,predDF,pd.DataFrame(y_out)],axis=1)

df2.columns = list(df.columns)+['pred','ideal']
print(df2)

valid_loss = np.array([i["valid_loss"] for i in network.train_history_])

# Create a Pandas Excel writer using XlsxWriter as the engine.
writer = pd.ExcelWriter('naval_regression_theano_multioutput.xlsx', engine='xlsxwriter')

# Convert the dataframe to an XlsxWriter Excel object.
df2.to_excel(writer, sheet_name='Sheet1')

# Close the Pandas Excel writer and output the Excel file.
writer.save()
