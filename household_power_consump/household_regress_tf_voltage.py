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
	# Fit command needs 64bit
    #if target_type in (np.int64, np.int32):
        # Classification
    #    return df.as_matrix(result).astype(np.float32),df.as_matrix([target]).astype(np.int32)
    #else:
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
    first_hidden_layer = tf.contrib.layers.relu(features, 64)
    second_hidden_layer = tf.contrib.layers.relu(first_hidden_layer, 512)
    third_hidden_layer = tf.contrib.layers.relu(second_hidden_layer, 1024)
    fourth_hidden_layer = tf.contrib.layers.relu(third_hidden_layer, 128)
    # Connect the output layer to second hidden layer (no activation fn)
    output_layer = tf.contrib.layers.linear(fourth_hidden_layer, 1)
 
    # Reshape output layer to 1-dim Tensor to return predictions
    #predictions = tf.reshape(output_layer, [-1,2])
    predictions = tf.reshape(output_layer,[-1])
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

x_out,y_out = to_xy(df,'Voltage')
## ABOVE HERE CHANGED
#x_out,y_out = to_xy(df,['Sub_metering_1','Sub_metering_2','Sub_metering_3'])

X_train, x_test, y_train, y_test = train_test_split(
    x_out, y_out, test_size=0.20, random_state=42)


print(y_train.shape)
print(X_train.shape)
    
	
# Get/clear a directory to store the neural network to
model_dir = get_model_dir('household',True)

model_params = {"learning_rate": LEARNING_RATE}
nn = tf.contrib.learn.Estimator(
    model_fn=model_fn, params=model_params)

validation_monitor = tf.contrib.learn.monitors.ValidationMonitor(
    x_test,
    y_test,
    every_n_steps=500,
    early_stopping_metric="loss",
    early_stopping_metric_minimize=True,
    early_stopping_rounds=50)

# Need to set batch size
#100 Epochs

N,M = X_train.shape
num_epoch = 100
num_steps = int((N/BATCH_SIZE)*num_epoch)

print("Number of steps: " + str(num_steps))

nn.fit(X_train,
	y_train,
	steps=num_steps,
	batch_size=BATCH_SIZE,
	monitors=[validation_monitor])


ev = nn.evaluate(x=x_test, y=y_test, steps=1)
loss_score = ev["loss"]
print("Loss: %s" % loss_score)

numPartition = 8

new_xout = np.array_split(x_out,numPartition)
new_yout = np.array_split(y_out,numPartition)

new_df = np.array_split(df,numPartition)

for ji in range(0,numPartition):
    print('Starting to predict')
    pred = list(nn.predict(new_xout[ji], as_iterable=True))
    predDF = pd.DataFrame(pred)
    df2 = pd.concat([new_df[ji],predDF,pd.DataFrame(new_yout[ji])],axis=1)

    print('Starting to write to csv file')

    df2.to_csv("household_regression_tf" + str(ji) + ".csv", chunksize=1000)

#df2.columns = list(df.columns)+['pred','ideal']
#print(df2)

# Create a Pandas Excel writer using XlsxWriter as the engine.
#writer = pd.ExcelWriter('household_regression_tf.xlsx', engine='xlsxwriter', options={'constant_memory': True})

# Convert the dataframe to an XlsxWriter Excel object.
#df2.to_excel(writer, sheet_name='Sheet1')

# Close the Pandas Excel writer and output the Excel file.
#writer.save()
