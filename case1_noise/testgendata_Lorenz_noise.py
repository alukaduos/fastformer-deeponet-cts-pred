from dysts.flows import Lorenz
#from dysts.analysis import find_lyapunov_exponents as lyaexp
import matplotlib.pyplot as plt
import matplotlib
import torch
import numpy as np
matplotlib.use( 'Agg')
from joblib import dump, load
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
            temp_ind_pred_begin = temp_mat[-1, -1] + self.gap_topred +1
            temp_ind_pred_end = temp_ind_pred_begin + self.num_pred + 1
            ind_mat_list.append(temp_mat)
            ind_pred_list.append(list(range(temp_ind_pred_begin, temp_ind_pred_end, 1)))
        return ind_mat_list,ind_pred_list

class Generate_UTS_Noise:
    """
    dyns        = dynamic system
    dim_emb     = ...
    time_delay  = time delay
    num_periods = how many periods to generate
    pts_per_period = how many points per period
    dur_predict = the time range limit of prediction
    """
    def __init__(self,dyns,num_period,pts_period,dim_emb=None,time_delay=None,dur_predict=None,noise_strength = 0.2):
        self.dim_emb = dim_emb
        self.time_delay = time_delay
        self.num_period = num_period
        self.pts_period = pts_period
        self.dur_predict = dur_predict
        self.dyns = dyns
        self.noise_strength = noise_strength
    def gettraindata(self):
        num_discard = 1
        # lya = lyaexp(self.dyns,num_simulate,pts_per_period = self.pts_period)
        lya = self.dyns.maximum_lyapunov_estimated
        num_simulate = self.num_period * self.pts_period + num_discard
        timestamp,sol = self.dyns.make_trajectory(num_simulate,pts_per_period = self.pts_period,return_times = True)
        deltatime = timestamp[1] - timestamp[0] #sample's delta t
        timelen_period = self.pts_period * deltatime #time duration of a oscillation period
        timestamp = timestamp[num_discard:]
        lya_scale = int((1 / lya) // deltatime)
        ref_scale = 1*lya_scale
        if self.dim_emb is None:
            self.dim_emb = int(np.floor(np.sqrt(ref_scale))*2)
        dim_delay = int(ref_scale//self.dim_emb)
        if self.time_delay is None:
            self.time_delay = dim_delay*deltatime
        #self.dur_predict = self.dim_emb * dim_delay*deltatime
        dynseries = sol[num_discard:,0]
        dynseries_noise = dynseries+np.random.normal(loc=0,scale=self.noise_strength*np.std(dynseries),size=dynseries.shape)
        #dim_delay = int(np.floor(self.time_delay / deltatime))
        fromind,endind = 0,dynseries.shape[0]-1
        #num_eleinrow = dim_delay
        #dim_emb = self.dim_emb
        gap_topred = 1
        if self.dur_predict is not None:
            num_pred = int(self.dur_predict // deltatime)
        else:
            num_pred = self.dim_emb*dim_delay-1
        sampleconti = True
        """
        ind_mat_list used to store index of segments, each element is corresponding to a segments, 
        it is a matrix whose rows are indices of embedded vectors
        ind_pred_list used to store index of predictions, each element is corresponding to a segments              
        """
        if not sampleconti:
            ind_mat_list, ind_pred_list = Indx_Create_UTS(fromind,endind,dim_delay,self.dim_emb,gap_topred,num_pred).indreshape()
        else:
            ind_mat_list, ind_pred_list = Indx_Create_UTS(fromind,endind,dim_delay,self.dim_emb,gap_topred,num_pred).indreshapeconti()

        num_samples = len(ind_pred_list)
        train_input = []
        train_mat = []
        train_target = []
        train_input_noise = []
        train_mat_noise = []
        train_target_noise = []
        for idx, idx_mat in enumerate(ind_mat_list[0:-2]):
            temp_idx_mat = np.array(idx_mat).transpose().flatten()#natural order
            temp_idx_pred = ind_pred_list[idx]#natural order
            train_input.append(dynseries[temp_idx_mat])#natural order
            train_mat.append(dynseries[temp_idx_mat].reshape(-1,dim_delay).transpose())#embedded order
            train_target.append(dynseries[temp_idx_pred])#natural order
            train_input_noise.append(dynseries_noise[temp_idx_mat]) #natural order
            train_mat_noise.append(dynseries_noise[temp_idx_mat].reshape(-1,dim_delay).transpose()) #embedded order
            train_target_noise.append(dynseries_noise[temp_idx_pred]) #natural order
        outputargs = {"train_input": train_input, "train_mat": train_mat, "train_target": train_target,
                      "ind_mat_list": ind_mat_list, "ind_pred_list": ind_pred_list,
                      "lya_scale":lya_scale, "timestamp":timestamp, "dim_emb": self.dim_emb,
                      "train_input_noise": train_input_noise, "train_mat_noise": train_mat_noise, "train_target_noise": train_target_noise}
        return outputargs

class ResultDisplay:
    """
    outputsargs includes train_input,train_mat,train_target,
    this is to show several examples of input seq vs target seq
    """
    def __init__(self,outputsargs,sampleconti=True,row_figs=3,col_figs=3,save_f_names=None):
        self.outputsargs = outputsargs
        self.sampleconti = sampleconti
        self.row_figs = row_figs
        self.col_figs = col_figs
        self.save_f_names = save_f_names
    def dodisplay(self):
        trainset = self.outputsargs["train_input"]
        train_mat = self.outputsargs["train_mat"]
        targetset = self.outputsargs["train_target"]
        ind_mat_list = self.outputsargs["ind_mat_list"]
        ind_pred_list = self.outputsargs["ind_pred_list"]
        lya_scale = self.outputsargs["lya_scale"]
        timestamp = self.outputsargs["timestamp"]
        dim_embs = self.outputsargs["dim_emb"]
        train_input_noise = self.outputsargs["train_input_noise"]
        train_mat_noise = self.outputsargs["train_mat_noise"]
        train_target_noise = self.outputsargs["train_target_noise"]
        print(f"The lya_scale is {lya_scale}.")
        sampleconti = self.sampleconti
        plt.figure(1,figsize=(17,8))
        print("Fig. 1: Train set, inputs and outputs are contaminated by noise")
        for idxfig in list(range(0,self.row_figs*self.col_figs,1)):
            plt.subplot(self.row_figs,self.col_figs,idxfig+1)
            if not sampleconti:
                figgap = 1
            else:
                figgap = len(ind_pred_list[0])
            for idxline in list(range(0,train_mat[idxfig].shape[0],1)):
                if idxline == 0:
                    markersizes = 5
                else:
                    markersizes = 2
                #plt.plot(np.array(ind_mat_list[idxfig*figgap])[idxline,:],np.array(train_mat[idxfig*figgap])[idxline,:],'o',markersize=markersizes)
                plt.plot(np.array(ind_mat_list[idxfig*figgap])[idxline,:],np.array(train_mat_noise[idxfig*figgap])[idxline,:],'o',markersize=markersizes)
            plt.plot(ind_pred_list[idxfig*figgap],train_target_noise[idxfig*figgap],'r*--')
        plt.savefig(self.save_f_names[0], dpi=300, bbox_inches='tight')

        plt.figure(2,figsize=(8,8))
        print("Fig. 2: Test set, inputs are contaminated by noise, outputs are noisy free")
        for idxfig in list(range(8,9,1)):
            plt.subplot(1,1,1)
            for idxline in list(range(0,train_mat[idxfig].shape[0],1)):
                if idxline == 0:
                    markersizes = 8
                else:
                    markersizes = 5
                #plt.plot(np.array(ind_mat_list[idxfig])[idxline,:],np.array(train_mat[idxfig])[idxline,:],'o',markersize=markersizes)
                plt.plot(np.array(ind_mat_list[idxfig*figgap])[idxline,:],np.array(train_mat_noise[idxfig*figgap])[idxline,:],'o',markersize=markersizes)
            plt.plot(ind_pred_list[idxfig*figgap],targetset[idxfig*figgap],'r*--',markersize=markersizes)
        plt.savefig(self.save_f_names[1], dpi=300, bbox_inches='tight')
        #plt.figure(3)
        print("Fig. 3: embedding vectors are aligned")
        for idxfig in list(range(8, 9, 1)):
            figg, axess = plt.subplots(1,1,figsize=(8, 8))
            for idxline in list(range(0, train_mat[idxfig].shape[0], 1)):
                if idxline == 0:
                    markersizes = 5
                else:
                    markersizes = 2
                axess.plot(np.array(ind_mat_list[idxfig*figgap])[idxline, :]-idxline, np.array(train_mat_noise[idxfig*figgap])[idxline, :], 'o--',
                         markersize=markersizes)
            axess.set_xticks(np.array(ind_mat_list[idxfig*figgap])[0, :])
            axess.set_xticklabels(list(range(1, dim_embs + 1, 1)))
        plt.savefig(self.save_f_names[2], dpi=300, bbox_inches='tight')
        plt.show()

            #axess.set_xticklabels(str(np.array(ind_mat_list[idxfig])[0, :] - np.array(ind_mat_list[idxfig])[0, :].min()))

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

"""some key constants can put here for other .py using"""
#noise_strength = 0.1

"""end"""

if __name__ == '__main__':
    #dim_embs = 16
    noise_strength = 0.10
    #print("Noise Strength is:",noise_strength)
    save_name = genjname(str(noise_strength))
    save_f_name = (genfname(str(noise_strength)+"in_out_examples"),
                   genfname(str(noise_strength)+"one_example"),
                   genfname(str(noise_strength)+"embvecs"))

    if False:
        outputsargs = Generate_UTS_Noise(Lorenz(),400,50,noise_strength=noise_strength).gettraindata()
        dump(outputsargs,save_name)
    else:
        outputsargs = load(save_name)
    ResultDisplay(outputsargs,sampleconti=True,row_figs=3,col_figs=3,save_f_names=save_f_name).dodisplay()



