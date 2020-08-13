# the required python libraries imported
import bnpy
import multiprocessing as mp
import pandas as pd
import numpy as np
import os
import time
from glob import glob

batch_size = 5
window_size_in_batches = 5
windows = []

def run_bnp_anomaly(mppack):
#     type(w[0][0][0])
# <class 'pandas.core.series.Series'>
# type(w[0][0][1])
    windows = mppack[0]
    batch_size = mppack[1]
    window_size_in_batches = mppack[2]
    data_set = mppack[3]
    G = mppack[4]

    wds = len(windows)
    gamma = 1.0
    sF = 1.0
    K = 25  # Initialize K component - this value places a max K the model can develop
    nLap = 20
    ds = data_set[0] + "." + data_set[1]

    iname='randexamples'
    opath = f'/tmp/bnp-anomaly/coldstart/{ds}/b0'  # Dynamic output path according to batch
    ll = [np.nan] * window_size_in_batches
    
    data_df = pd.DataFrame()
    calc_df = pd.DataFrame()
    
    
    for ii, window in enumerate(windows):

        df_index = window[0]
        xdata_data = window[1]

        if ii % 5 == 0:
            print("XXX" + str(data_set)+ " " + str(ii)+"/"+str(wds))

        warm_start_model, warm_info_dict = bnpy.run(
            xdata_data, 'DPMixtureModel', 'DiagGauss', 'memoVB',
            output_path=opath,
            nLap=nLap, nTask=1, nBatch=window_size_in_batches, convergeThr=0.0001,
            gamma0=gamma, sF=sF, ECovMat='eye',
            K=K, 
            moves='birth,merge,delete,shuffle',
            initname=iname,
            ts=True, debug=False, verbose=0, G=G)
        
        iname=warm_info_dict['task_output_path']
        opath = f'/tmp/bnp-anomaly/warmstart/{ds}/b{ii +  1}'

        batch = xdata_data.make_subset(list(range(batch_size * window_size_in_batches - batch_size, batch_size * window_size_in_batches)))

        LP = warm_start_model.calc_local_params(batch)
        SS = warm_start_model.get_global_suff_stats(batch, LP)
        LL = warm_start_model.calcLogLikCollapsedSamplerState(SS)

        ll.pop(0)
        ll.append(LL)
        ll_normed = [i/sum(ll) for i in ll]
        entropy = -sum([i*np.log(i) for i in ll_normed])

        index = df_index[-1:]
        x = df_index[len(df_index) - batch_size:]
        data_df = data_df.append({'x':x, 'y':batch.X.reshape((batch_size,))}, ignore_index=True)
        calc_df = calc_df.append({'index':index.iloc[0], 'LL':LL, 'entropy':entropy}, ignore_index=True)
        print('holdup')
#    results_df.set_index('index', inplace=True)
#    data_df.set_index('index', inplace=True)
    ds = data_set[0] + "." + data_set[1]
    data_n = "data/test_output/" + ds + "_data" + str(G) +  "_bs" + str(batch_size) \
                + "_wsib" + str(window_size_in_batches) + ".csv"
    calc_n = "data/test_output/" + ds + "_calc" + str(G) +  "_bs" + str(batch_size) \
                + "_wsib" + str(window_size_in_batches) + ".csv"
    data_df.to_csv(data_n)
    results_df.to_csv(calc_n)
    print("XXX wrote " + data_n)
    print("XXX wrote " + calc_n)
    return 0

G=0
test_data_dir = "data/test/"
test_data_files = sorted(glob(test_data_dir + '/ds01*.*.csv'))
test_data_names = [i.split("/")[2].split(".")[0:2] for i in test_data_files]
data = [pd.read_csv(i) for i in test_data_files]

for d, df in enumerate(data):
    win = []
    i = 0
    while i * batch_size <= (len(df) - window_size_in_batches * batch_size):
        w= df[i * batch_size:i * batch_size + window_size_in_batches * batch_size]
        y = w.drop(columns=['Unnamed: 0'])
        yy = bnpy.data.XData.from_dataframe(y)
        x = w['Unnamed: 0']
        win.append([x, yy])
        i += 1
    windows.append([win, batch_size, window_size_in_batches, test_data_names[d], G])


# pool =  mp.Pool(mp.cpu_count())
# results = pool.map(run_bnp_anomaly, windows)
# pool.close()

for w in windows:
    run_bnp_anomaly(w)