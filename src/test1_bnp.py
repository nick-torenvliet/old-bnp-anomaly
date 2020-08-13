# the required python libraries imported
import bnpy
import multiprocessing as mp
import pandas as pd
import numpy as np
import os
import time

batch_size = 5
window_size_in_batches = 5
windows = []

def run_bnp_anomaly(mppack):
    windows = mppack[0]
    batch_size = mppack[1]
    window_size_in_batches = mppack[2]
    data_set = mppack[3]
    wds = len(windows)
    gamma = 1.0
    sF = 1.0
    K = 25  # Initialize K component - this value places a max K the model can develop
    nLap = 20
    iname='randexamples'
    opath = f'/tmp/bnp-anomaly/coldstart/1{data_set}/b0'  # Dynamic output path according to batch
    ll = [np.nan] * window_size_in_batches
    
    data_df = pd.DataFrame(columns =['index', 'data'])
    results_df = pd.DataFrame()
    
    for ii, window in enumerate(windows):
        if ii % 5 == 0:
            print("XXX" + str(data_set)+ " " + str(ii)+"/"+str(wds))

        warm_start_model, warm_info_dict = bnpy.run(
            window, 'DPMixtureModel', 'DiagGauss', 'memoVB',
            output_path=opath,
            nLap=nLap, nTask=1, nBatch=window_size_in_batches, convergeThr=0.0001,
            gamma0=gamma, sF=sF, ECovMat='eye',
            K=K, 
            moves='birth,merge,delete,shuffle',
            initname=iname,
            ts=True, debug=False, verbose=0, G=1)
        
        iname=warm_info_dict['task_output_path']
        opath = f'/tmp/bnp-anomaly/warmstart/1{data_set}/b{ii +  1}'

        batch = window.make_subset(list(range(batch_size * window_size_in_batches - batch_size, batch_size * window_size_in_batches)))

        LP = warm_start_model.calc_local_params(batch)
        SS = warm_start_model.get_global_suff_stats(batch, LP)
        LL = warm_start_model.calcLogLikCollapsedSamplerState(SS)

        ll.pop(0)
        ll.append(LL)
        ll_normed = [i/sum(ll) for i in ll]
        entropy = -sum([i*np.log(i) for i in ll_normed])

        K_resp = np.mean(LP["resp"], axis=0)
        x_window  = SS.x
        xx_window = SS.xx

        x_window = np.vstack(x_window)
        xx_window = np.vstack(xx_window)

        index = int(ii * batch_size) + window_size_in_batches * batch_size - 1
        x_window = x_window.flatten()
        x_window = x_window[x_window >1]
        xx_window = xx_window.flatten()
        y = (window.X[-batch_size:])[0::1]
        y = y.flatten()
        x = list(range(ii * batch_size, ii * batch_size + window_size_in_batches * batch_size))[-batch_size:][0::1]
        ddf = pd.DataFrame(columns =['index', 'data'])
        ddf['data'] = np.array(y)
        ddf['index'] = np.array(x)
        data_df = data_df.append(ddf, ignore_index=True)
        results_df = results_df.append({'index':index, 'LL':LL, 'entropy':entropy}, ignore_index=True)
        print('holdup')
    results_df.set_index('index', inplace=True)
    data_df.set_index('index', inplace=True)
    return [data_df, results_df, data_set]

data_sets = ["./data/test/ds0.csv",
             "./data/test/ds1.csv",
             "./data/test/ds2.csv",
             "./data/test/ds3.csv",
             "./data/test/ds4.csv",
             "./data/test/ds5.csv",
             "./data/test/ds6.csv",
             "./data/test/ds7.csv"
            ]
data = [pd.read_csv(i, usecols=['0']) for i in data_sets]

for d, df in enumerate(data):
    win = []
    i = 0
    while i * batch_size < (len(df) - window_size_in_batches * batch_size):
        w = df[i * batch_size:i * batch_size + window_size_in_batches * batch_size]
        win.append(bnpy.data.XData.from_dataframe(w))
        i += 1
    windows.append([win, batch_size, window_size_in_batches, d])

alg = "G1"
pool =  mp.Pool(mp.cpu_count())
results = pool.map(run_bnp_anomaly, windows)
pool.close()
for results in results:
    data = results[0]
    calc = results[1]
    dset = results[2]
    data.to_csv("data/test_output/dataset" + str(dset) + "_data_test_alg" + alg +  "_bs" + str(batch_size) 
                + "_wsib" + str(window_size_in_batches) + ".csv")
    calc.to_csv("data/test_output/dataset" + str(dset) + "_results_test_alg" + alg +  "_bs" + str(batch_size) 
                + "_wsib" + str(window_size_in_batches) + ".csv")

    