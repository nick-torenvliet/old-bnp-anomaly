# the required python libraries imported
import bnpy
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import os

data_start = 0
data_init_size = 1800
batch_size = 200
batchnum = int(data_init_size/batch_size)

all_data = pd.read_csv('../data/anomaly0245.csv')
all_data.drop(all_data.columns[0], inplace=True, axis=1)

init_data = all_data.head(data_init_size)
init_data = bnpy.data.XData.from_dataframe(init_data)

data_set = bnpy.data.XData.from_dataframe(all_data)

batches = []
i = 0 
while i < len(all_data)- batch_size:
    df = all_data.iloc[i:i + batch_size]
    batches.append(bnpy.data.XData.from_dataframe(df))
    i += batch_size

anomalies = [78, 98, 99, 104, 105, 106, 122, 124, 127, 128, 129, 130, 131, 156, 157, 158, 159, 160, 161, 162, 163, 164, 165, 166]
cleans = [11, 12, 18, 19, 30, 33, 40, 52, 58, 59, 60, 61, 74, 79, 80, 81, 186, 187, 188, 189, 190, 191, 192, 193]

# Setup the placekeeping and initilizing variables
chain = 0
x, eng_val, states, num_states = [], [], [], []
i = 0
print(i)

# Initialize bnpy model and do initial training
# *DiagGauss* observation model
gamma = 1.0
sF = 1.0
K = 25  # Initialize K component
nLap = 10

cold_start_model, cold_info_dict = bnpy.run(
    init_data, 'DPMixtureModel', 'DiagGauss', 'memoVB',
    output_path='/tmp/AsteriskK8/coldstart-K=10/',
    nLap=nLap, nTask=1, nBatch=batchnum, convergeThr=0.0001,
    gamma0=gamma, sF=sF, ECovMat='eye',
    K=K, initname='randexamplesbydist', ts=True)

# Get the intial graphing data
y = np.squeeze(init_data.X)
x = list(range(0, len(init_data.X)))
x_batches = []
x_batch_post = []
x_batch_pre = []
K_model = []
K_states = []
index = []

warm_start_model = cold_start_model
warm_info_dict = cold_info_dict

for i in range(int(data_init_size/batch_size), len(batches)):
   
    # Shift the dataset to include new incoming data   
    new_dataset = data_set.make_subset(example_id_list = list(range(i * batch_size - data_init_size, i * batch_size)))
    
    # Check sufficient statistics on the new batch with the previously learned model 
    LPanomaly = []
    SSanomaly = []
    LP = warm_start_model.calc_local_params(batches[i])
    LPanomaly.append(LP)  # Calculation of responsibility, needed for next step
    SSanomaly.append(warm_start_model.get_global_suff_stats(batches[i], LP))  # Calculation of SS for new data
    x_batch_pre = []
    xx_batch_pre = []
    for key in SSanomaly:
        x_batch_pre.append(key.x)
        xx_batch_pre.append(key.xx)
    x_batch_pre = np.vstack(x_batch_pre)
    xx_batch_pre = np.vstack(xx_batch_pre)
   
    warm_start_model, warm_info_dict = bnpy.run(
        new_dataset, 'DPMixtureModel', 'DiagGauss', 'memoVB',
        output_path='/tmp/AsteriskK8/warmstart-K=10/',
        nLap=nLap, nTask=1, nBatch=batchnum, convergeThr=0.0001,
        gamma0=gamma, sF=sF, ECovMat='eye',
        K=K, initname=cold_info_dict['task_output_path'], ts=True)#     trained_model, trained_dict = bnpy.run(
       
    # Check sufficient statistics on the new batch with the newly learned model 
    LPanomaly = []
    SSanomaly = []
    LP = warm_start_model.calc_local_params(batches[i])
    LPanomaly.append(LP)  # Calculation of responsibility, needed for next step
    SSanomaly.append(warm_start_model.get_global_suff_stats(batches[i], LP))  # Calculation of SS for new data
    x_batch_post = []
    xx_batch_post = []
    K_model = []
    K_states = []
    for key in SSanomaly:
        x_batch_post.append(key.x)
        xx_batch_post.append(key.xx)    
        K_model.append(key.K)
    x_batch_post = np.vstack(x_batch_post)
    xx_batch_post = np.vstack(xx_batch_post)
    K_model = np.vstack(K_model)
    
    index = int(i * batch_size)
    x_batch_pre = np.squeeze(np.squeeze(x_batch_pre))
    x_batch_post = np.squeeze(np.squeeze(x_batch_post))
    K_model = np.sum(x_batch_post > 1)
    y = np.squeeze(batches[i].X)
    x = list(range(i*len(batches[i].X), i*len(batches[i].X) + len(batches[i].X)))
