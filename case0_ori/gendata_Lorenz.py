from dysts.flows import Lorenz
from dysts.flows import MackeyGlass
from dysts.analysis import find_lyapunov_exponents as lyaexp
import matplotlib.pyplot as plt
import matplotlib
import torch
import numpy as np
import csv
import pandas as pd
matplotlib.use('Agg')
from joblib import dump, load
import time
from datetime import datetime
import os
from pathlib import Path
def genjname(specnote=None):
    file_path = os.path.dirname(os.path.abspath(__file__))
    file_name = os.path.basename(__file__)
    file_full = (os.path.join(file_path, file_name))
    file_without_extension = Path(file_full).stem
    # spec_flag = "noise"
    save_name = os.path.join(file_path, file_without_extension + "_" + specnote + ".joblib")
    return save_name

def genfname(specnote=None):
    file_path = os.path.dirname(os.path.abspath(__file__))
    file_name = os.path.basename(__file__)
    file_full = (os.path.join(file_path, file_name))
    file_without_extension = Path(file_full).stem
    save_name = os.path.join(file_path, file_without_extension + "_" + specnote + ".pdf")
    return save_name

class Indx_Create_UTS:
    """
    fromind    = where extraction begins
    endind     = where extraction ends
    num_eleinrow = delay lag
    dim_emb    = ...
    gap_topred = lags between the last record and the first prediction
    num_pred   = length of prediction
    example:
    ind_mat_list,ind_pred_list = Indx_Create_UTS(3,289,5,10,0,10).indreshape()
    """
    def __init__(self,fromind,endind,num_eleinrow,dim_emb,gap_topred,num_pred):
        self.fromind = fromind
        self.endind = endind
        self.num_eleinrow = num_eleinrow # = delay lag
        self.dim_emb = dim_emb
        self.gap_topred = gap_topred
        self.num_pred = num_pred
    def indreshape(self):
        ind_ori_0 = list(range(self.fromind,self.endind,1))
        temp_end = int(-(len(ind_ori_0)%(self.dim_emb*self.num_eleinrow)))
        ind_ori = ind_ori_0[:temp_end]
        ind_mat = np.array(ind_ori).reshape(-1,self.num_eleinrow).transpose()
        num_rec = int((len(ind_ori)-self.num_pred)//(self.dim_emb*self.num_eleinrow))
        ind_mat_list = []
        ind_pred_list = []
        for idx in range(num_rec):
            temp_mat = ind_mat[:,idx*self.dim_emb:(idx+1)*self.dim_emb]
            temp_ind_pred_begin = temp_mat[-1,-1]+self.gap_topred
            temp_ind_pred_end = temp_ind_pred_begin + self.num_pred+1
            ind_mat_list.append(temp_mat)
            ind_pred_list.append(list(range(temp_ind_pred_begin,temp_ind_pred_end,1)))
        return ind_mat_list,ind_pred_list
    def indreshapeconti(self):
        ind_ori_0 = list(range(self.fromind,self.endind,1))
        temp_end = int(-(len(ind_ori_0) % (self.dim_emb * self.num_eleinrow)))
        ind_ori = ind_ori_0[:temp_end]
        ind_mat_list = []
        ind_pred_list = []
        for idx in range(len(ind_ori)-2*self.dim_emb*self.num_eleinrow):
            temp_mat = np.array(ind_ori[idx:idx+self.dim_emb*self.num_eleinrow]).reshape(-1,self.num_eleinrow).transpose()
            temp_ind_pred_begin = temp_mat[-1, -1] + self.gap_topred
            temp_ind_pred_end = temp_ind_pred_begin + self.num_pred + 1
            ind_mat_list.append(temp_mat)
            ind_pred_list.append(list(range(temp_ind_pred_begin, temp_ind_pred_end, 1)))
        return ind_mat_list,ind_pred_list



class Generate_UTS:
    """
    dyns        = dynamic system
    dim_emb     = ...
    time_delay  = time delay
    num_periods = how many periods to generate
    pts_per_period = how many points per period
    dur_predict = the time range limit of prediction
    """
    def __init__(self,dyns,num_period,pts_period,dim_emb=None,time_delay=None,dur_predict=None):
        self.dim_emb = dim_emb
        self.time_delay = time_delay
        self.num_period = num_period
        self.pts_period = pts_period
        self.dur_predict = dur_predict
        self.dyns = dyns

    def gettraindata(self):
        num_discard = 1
        # lya = lyaexp(self.dyns,num_simulate,pts_per_period = self.pts_period)
        lya = self.dyns.maximum_lyapunov_estimated
        num_simulate = self.num_period * self.pts_period + num_discard
        flag_regen = False
        if flag_regen:
            timestamp,sol = self.dyns.make_trajectory(num_simulate,pts_per_period = self.pts_period,return_times = True)
            temp_lorenz_ori = np.hstack((timestamp.reshape(len(timestamp),1),sol))
            pd.DataFrame(temp_lorenz_ori).to_csv("lorenz_ori.csv")
            #pd.DataFrame(temp_lorenz_ori).to_csv("MG_ori.csv")
        else:
            temp_lorenz_ori = pd.read_csv("lorenz_ori.csv")
            #temp_lorenz_ori = pd.read_csv("MG_ori.csv")
            timestamp = np.array(temp_lorenz_ori.iloc[:,1]).astype(np.float32).reshape(-1,1)
            sol = np.array(temp_lorenz_ori.iloc[:,2:5]).astype(np.float32)
        #timestamp = np.array(timestamp[num_discard:],dtype=float).reshape(-1,1)

        #deltatime = timestamp[1,0] - timestamp[0,0] #sample's delta t
        deltatime = timestamp[1] - timestamp[0] #sample's delta t

        lya_scale = int((1 / lya) // deltatime)
        timelen_period = self.pts_period * deltatime #time duration of a oscillation period
        ref_scale = 1 * lya_scale



        if self.dim_emb is None:
            self.dim_emb = int(np.floor(np.sqrt(ref_scale))*2)
        dim_delay = int(ref_scale//self.dim_emb)
        if self.time_delay is None:
            self.time_delay = dim_delay*deltatime
        #self.dur_predict = self.dim_emb * dim_delay*deltatime
        dynseries = sol[num_discard:,0]
        #dim_delay = int(np.floor(self.time_delay / deltatime))
        fromind,endind = 0,dynseries.shape[0]-1
        #num_eleinrow = dim_delay
        #dim_emb = self.dim_emb
        gap_topred = 1
        if self.dur_predict is not None:
            num_pred = int(self.dur_predict // deltatime)
        else:
            num_pred = self.dim_emb*dim_delay-1
        with open('lormkattr.txt','a',encoding='utf-8') as f:
            currentdt = datetime.now().replace(microsecond=0)
            f.write(f"Lorenz, without noise, time={currentdt}, deltatime={str(deltatime[0])},lya={lya:.4f}, period_duration = {str(self.pts_period * deltatime)}, "
                    f"lya_scale={lya_scale}, dim_emb={self.dim_emb}, dim_delay = {dim_delay}, time_delay={str(self.time_delay[0])}\n")
        sampleconti = False
        if sampleconti:
            ind_mat_list, ind_pred_list = Indx_Create_UTS(fromind,endind,dim_delay,self.dim_emb,gap_topred,num_pred).indreshape()
        else:
            ind_mat_list, ind_pred_list = Indx_Create_UTS(fromind,endind,dim_delay,self.dim_emb,gap_topred,num_pred).indreshapeconti()

        num_samples = len(ind_pred_list)
        train_input = []
        train_mat = []
        train_target = []
        for idx, idx_mat in enumerate(ind_mat_list[0:-2]):
            temp_idx_mat = np.array(idx_mat).transpose().flatten()
            temp_idx_pred = ind_pred_list[idx]
            train_input.append(dynseries[temp_idx_mat])
            train_mat.append(dynseries[temp_idx_mat].reshape(-1,dim_delay).transpose())
            train_target.append(dynseries[temp_idx_pred])
        outputargs = {"train_input": train_input, "train_mat": train_mat, "train_target": train_target,
                      "ind_mat_list": ind_mat_list, "ind_pred_list": ind_pred_list,
                      "lya_scale":lya_scale, "timestamp":timestamp, "dim_emb": self.dim_emb}
        return outputargs

if __name__ == '__main__':
    
    save_name = genjname("ori")
    if True:
        outputsargs = Generate_UTS(Lorenz(),400,50).gettraindata()
        dump(outputsargs,save_name)
    else:
        outputsargs = load(save_name)
    trainset = outputsargs["train_input"]; train_mat=outputsargs["train_mat"]; targetset=outputsargs["train_target"]
    ind_mat_list = outputsargs["ind_mat_list"]; ind_pred_list = outputsargs["ind_pred_list"]
    lya_scale = outputsargs["lya_scale"]; timestamp = outputsargs["timestamp"]; dim_embs = outputsargs["dim_emb"]
    print(f"The lya_scale is {lya_scale}.")
    dim_delay, dim_emb = ind_mat_list[0].shape


    fig_name_1 = genfname("in_out_examples")
    fig_name_2 = genfname("one_example")
    fig_name_3 = genfname("embvecs")

    plt.figure(1,figsize=(17, 8))
    for idxfig in list(range(0,9,1)):
        plt.subplot(3,3,idxfig+1)
        for idxline in list(range(0,train_mat[idxfig].shape[0],1)):
            if idxline == 0:
                markersizes = 5
            else:
                markersizes = 2
            plt.plot(np.array(ind_mat_list[idxfig*dim_delay*dim_emb])[idxline,:],np.array(train_mat[idxfig*dim_delay*dim_emb])[idxline,:],'o',markersize=markersizes)
        plt.plot(ind_pred_list[idxfig*dim_delay*dim_emb],targetset[idxfig*dim_delay*dim_emb],'r*--')
    plt.savefig(fig_name_1, dpi=300, bbox_inches='tight')


    plt.figure(2,figsize=(8, 8))

    for idxfig in list(range(8,9,1)):
        plt.subplot(1,1,1)
        idxfig_ext = idxfig * ind_mat_list[0].size
        for idxline in list(range(0,train_mat[idxfig].shape[0],1)):
            if idxline == 0:
                markersizes = 5
            else:
                markersizes = 2
            plt.plot(np.array(ind_mat_list[idxfig_ext])[idxline,:],np.array(train_mat[idxfig_ext])[idxline,:],'o',markersize=markersizes)
        plt.plot(ind_pred_list[idxfig_ext],targetset[idxfig_ext],'r*--')
    plt.savefig(fig_name_2, dpi=300, bbox_inches='tight')

    #plt.figure(3)
    for idxfig in list(range(8, 9, 1)):
        figg, axess = plt.subplots(1,1,figsize=(8, 8))
        for idxline in list(range(0, train_mat[idxfig].shape[0], 1)):
            if idxline == 0:
                markersizes = 5
            else:
                markersizes = 2
            axess.plot(np.array(ind_mat_list[idxfig_ext])[idxline, :]-idxline, np.array(train_mat[idxfig_ext])[idxline, :], 'o--',
                     markersize=markersizes)

        axess.set_xticks(np.array(ind_mat_list[idxfig_ext])[0, :])
        axess.set_xticklabels(list(range(1,dim_embs+1,1)))
    plt.savefig(fig_name_3, dpi=300, bbox_inches='tight')
    plt.show()
        #plt.plot(ind_pred_list[idxfig], targetset[idxfig], '*')

    plt.figure(1)
    plt.show()
    #plt.ioff()
    plt.figure(2)
    plt.show()
    #plt.ioff()
    # plt.figure(3)
    # plt.show()
    #plt.ioff()


