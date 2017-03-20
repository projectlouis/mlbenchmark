import pandas as pd
import os
import shutil
import numpy as np
import tensorflow as tf
from sklearn import metrics
import tensorflow.contrib.learn as learn
#from scipy.stats import zscore
import matplotlib.pyplot as plt
from tensorflow.contrib.learn.python.learn.estimators import model_fn as model_fn_lib
from sklearn.model_selection import train_test_split

tf.logging.set_verbosity(tf.logging.INFO)

# Encode a numeric column as zscores
def encode_numeric_zscore(df,name,mean=None,sd=None):
    if mean is None:
        mean = df[name].mean()

    if sd is None:
        sd = df[name].std()

    df[name] = (df[name]-mean)/sd

# Convert all missing values in the specified column to the median
def missing_median(df, name):
    med = df[name].median()
    df[name] = df[name].fillna(med)

# Convert a Pandas dataframe to the x,y inputs that TensorFlow needs
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

# Regression chart, we will see more of this chart in the next class.
def chart_regression(pred,y):
    t = pd.DataFrame({'pred' : pred, 'y' : y_test.flatten()})
    t.sort_values(by=['y'],inplace=True)
    a = plt.plot(t['y'].tolist(),label='expected')
    b = plt.plot(t['pred'].tolist(),label='prediction')
    plt.ylabel('output')
    plt.legend()
    plt.show()
    
# Get a new directory to hold checkpoints from a neural network.  This allows the neural network to be
# loaded later.  If the erase param is set to true, the contents of the directory will be cleared.
def get_model_dir(name,erase):
    base_path = os.path.join(".","saved_models")
    model_dir = os.path.join(base_path,name)
    os.makedirs(model_dir,exist_ok=True)
    if erase and len(model_dir)>4 and os.path.isdir(model_dir):
        shutil.rmtree(model_dir,ignore_errors=True) # be careful, this deletes everything below the specified path
    return model_dir

def model_fn(features, targets, mode, params):
  # Connect the first hidden layer to input layer
  # (features) with relu activation
    first_hidden_layer = tf.contrib.layers.relu(features, 500) 
    # Connect the second hidden layer to first hidden layer with relu
    second_hidden_layer = tf.contrib.layers.relu(first_hidden_layer, 250)
    third_hidden_layer = tf.contrib.layers.relu(second_hidden_layer, 5)
    # Connect the output layer to second hidden layer (no activation fn)
    output_layer = tf.contrib.layers.linear(third_hidden_layer, 2)
 
    # Reshape output layer to 1-dim Tensor to return predictions
    #predictions = tf.reshape(output_layer, [-1,2])
    predictions = tf.reshape(output_layer,[-1,2])
    predictions_dict = {"prediction": predictions}

    # Calculate loss using mean squared error
    loss = tf.losses.mean_squared_error(targets, predictions)

    # Calculate root mean squared error as additional eval metric
    eval_metric_ops = {
        "rmse": tf.metrics.root_mean_squared_error(
            tf.cast(targets, tf.float64), predictions)
    }
  
    train_op = tf.contrib.layers.optimize_loss(
        loss=loss,
        global_step=tf.contrib.framework.get_global_step(),
        learning_rate=params["learning_rate"],
        optimizer="Adam")

    return model_fn_lib.ModelFnOps(
        mode=mode,
        predictions=predictions_dict,
        loss=loss,
        train_op=train_op,
        eval_metric_ops=eval_metric_ops)
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

#df = df.drop('gt_in_airT',1)

del df['gt_in_airT']

#x,y = to_xy(df,'GT_compre_coef')
#x,y = to_xy(df,'GT_turb_coef')
## ABOVE HERE CHANGED
x_out,y_out = to_xy(df,['GT_compre_coef','GT_turb_coef'])

X_train, x_test, y_train, y_test = train_test_split(
    x_out, y_out, test_size=0.20, random_state=42)

X_train = X_train[:,0:15]
x_test = x_test[:,0:15]
x_out = x_out[:,0:15]
	
# Get/clear a directory to store the neural network to
model_dir = get_model_dir('naval',True)

model_params = {"learning_rate": LEARNING_RATE}
nn = tf.contrib.learn.Estimator(
    model_fn=model_fn, model_dir=model_dir, params=model_params)

validation_monitor = tf.contrib.learn.monitors.ValidationMonitor(
    x_test,
    y_test,
    every_n_steps=100,
    early_stopping_metric="loss",
    early_stopping_metric_minimize=True,
    early_stopping_rounds=200)

	# Need to set batch size
nn.fit(X_train,
	y_train,
	steps=82700,
	batch_size=BATCH_SIZE,
	monitors=[validation_monitor])


ev = nn.evaluate(x=x_test, y=y_test, steps=1)
loss_score = ev["loss"]
print("Loss: %s" % loss_score)

pred = list(nn.predict(x, as_iterable=True))
predDF = pd.DataFrame(pred)
df2 = pd.concat([df,predDF,pd.DataFrame(y)],axis=1)

#df2.columns = list(df.columns)+['pred','ideal']
print(df2)

# Create a Pandas Excel writer using XlsxWriter as the engine.
writer = pd.ExcelWriter('naval_regression_tf_multi.xlsx', engine='xlsxwriter')

# Convert the dataframe to an XlsxWriter Excel object.
df2.to_excel(writer, sheet_name='Sheet1')

# Close the Pandas Excel writer and output the Excel file.
writer.save()
