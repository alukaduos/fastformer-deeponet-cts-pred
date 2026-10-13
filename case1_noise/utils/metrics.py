import numpy as np


def RSE(pred, true):
    return np.sqrt(np.sum((true - pred) ** 2)) / np.sqrt(np.sum((true - true.mean()) ** 2))


def CORR(pred, true):
    u = ((true - true.mean(0)) * (pred - pred.mean(0))).sum(0)
    d = np.sqrt(((true - true.mean(0)) ** 2 * (pred - pred.mean(0)) ** 2).sum(0))
    return (u / d).mean(-1)


def MAE(pred, true):
    return np.mean(np.abs(true - pred))



def MSE(pred, true):
    return np.mean((true - pred) ** 2)

def MSE_group(pred, true):
    return np.mean((pred - true)**2,axis=-1)
def NMSE_0(pred, true):
    return np.mean(MSE_group(pred, true)/np.std(true,axis=-1)**2)
def NMSE(pred, true):
    return MSE(pred, true)/np.std(true)**2

def RMSE(pred, true):
    return np.sqrt(MSE(pred, true))
def NRMSE_0(pred, true):
    return np.mean(np.sqrt(MSE_group(pred, true))/np.std(true,axis=-1))

def NRMSE(pred, true):
    return np.sqrt(MSE(pred, true))/np.std(true)
def normalize(vec):
    #vec = vec.flatten()
    #vec = vec - np.min(vec)+1
    #return (vec-vec.min()+1e-8)/(vec.max()-vec.min()+1e-8)
    return vec
def MAPE(pred, true):

    return np.mean(np.abs((normalize(true) - normalize(pred)) / normalize(true)))


def MSPE(pred, true):
    return np.mean(np.square((normalize(true) - normalize(pred)) / normalize(true)))


def metric(pred, true):
    mae = MAE(pred, true)
    mse = MSE(pred, true)
    rmse = RMSE(pred, true)
    mape = MAPE(pred, true)
    mspe = MSPE(pred, true)
    nmse = NMSE(pred, true)
    nrmse = NRMSE(pred, true)

    return mae, mse, rmse, mape, mspe, nmse, nrmse
def metric_vec(pred, true):
    idx_sample = 0;idx_time = -1
    mse_vec = np.mean((true - pred) ** 2, axis=idx_sample)
    mse_std = np.std((true - pred) ** 2, axis=idx_sample)
    return mse_vec, mse_std
"""
def metric_vec(pred, true):
    idx_sample = 0;idx_time = -1
    mae_vec = MAE(pred, true)
    mae_vec = np.mean(np.abs(pred - true), axis=idx_sample
    mse_std = np.std(np.mean((true - pred) ** 2, axis=idx_sample), axis=idx_time)
    rmse_std = np.std(np.sqrt(np.mean((true - pred) ** 2, axis=idx_sample)), axis=idx_time)
    mape_std = np.std(np.mean(np.abs((normalize(true) - normalize(pred)) / normalize(true)), axis=idx_sample),
                      axis=idx_time)
    mspe_std = np.std(np.mean(np.square((normalize(true) - normalize(pred)) / normalize(true)), axis=idx_sample),
                      axis=idx_time)
    nmse_std = np.std(np.mean((true - pred) ** 2, axis=idx_sample) / np.std(true, axis=idx_sample) ** 2, axis=idx_time)
    nrmse_std = np.std(np.sqrt(np.mean((true - pred) ** 2, axis=idx_sample)) / np.std(true, axis=idx_sample),
                       axis=idx_time)
    return mae_std, mse_std, rmse_std, mape_std, mspe_std, nmse_std, nrmse_std
"""
def metric_std(pred, true):
    """"""
    idx_sample = 0;idx_time = -1
    mae_std = np.std(np.mean(np.abs(pred - true),axis=idx_sample),axis=idx_time)
    mse_std = np.std(np.mean((true - pred)**2,axis=idx_sample),axis=idx_time)
    rmse_std = np.std(np.sqrt(np.mean((true - pred)**2,axis=idx_sample)),axis=idx_time)
    mape_std =  np.std(np.mean(np.abs((normalize(true) - normalize(pred)) / normalize(true)),axis=idx_sample),axis=idx_time)
    mspe_std = np.std(np.mean(np.square((normalize(true) - normalize(pred)) / normalize(true)),axis=idx_sample),axis=idx_time)
    nmse_std = np.std(np.mean((true - pred)**2,axis=idx_sample)/np.std(true,axis=idx_sample)**2,axis=idx_time)
    nrmse_std = np.std(np.sqrt(np.mean((true - pred)**2,axis=idx_sample))/np.std(true,axis=idx_sample),axis=idx_time)
    return mae_std, mse_std, rmse_std, mape_std, mspe_std, nmse_std, nrmse_std