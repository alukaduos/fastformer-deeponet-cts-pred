from dysts.flows import MackeyGlass
#from dysts.analysis import find_lyapunov_exponents as lyaexp
import matplotlib.pyplot as plt
import matplotlib
#import torch
import numpy as np
import os
from pathlib import Path
matplotlib.use( 'Agg')
from joblib import dump, load
#from gendata_Lorenz import Indx_Create_UTS as Indx_Create_UTS
from gendata_Lorenz import Generate_UTS as Generate_UTS

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

"""
MackeyGlass
"""

# class Indx_Create_UTS:
    # """
    # fromind    = where extraction begins
    # endind     = where extraction ends
    # num_eleinrow = delay lag
    # dim_emb    = ...
    # gap_topred = lags between the last record and the first prediction
    # num_pred   = length of prediction
    # example:
    # ind_mat_list,ind_pred_list = Indx_Create_UTS(3,289,5,10,0,10).indreshape()
    # """
# def __init__(self,fromind,endind,num_eleinrow,dim_emb,gap_topred,num_pred):
#     self.fromind = fromind
#     self.endind = endind
#     self.num_eleinrow = num_eleinrow # = delay lag
#     self.dim_emb = dim_emb
#     self.gap_topred = gap_topred
#     self.num_pred = num_pred
# def indreshape(self):
#     ind_ori_0 = list(range(self.fromind,self.endind,1))
#     temp_end = int(-(len(ind_ori_0)%(self.dim_emb*self.num_eleinrow)))
#     ind_ori = ind_ori_0[:temp_end]
#     ind_mat = np.array(ind_ori).reshape(-1,self.num_eleinrow).transpose()
#     num_rec = int((len(ind_ori)-self.num_pred)//(self.dim_emb*self.num_eleinrow))
#     ind_mat_list = []
#     ind_pred_list = []
#     for idx in range(num_rec):
#         temp_mat = ind_mat[:,idx*self.dim_emb:(idx+1)*self.dim_emb]
#         temp_ind_pred_begin = temp_mat[-1,-1]+self.gap_topred
#         temp_ind_pred_end = temp_ind_pred_begin + self.num_pred+1
#         ind_mat_list.append(temp_mat)
#         ind_pred_list.append(list(range(temp_ind_pred_begin,temp_ind_pred_end,1)))
#     return ind_mat_list,ind_pred_list
# def indreshapeconti(self):
#     ind_ori_0 = list(range(self.fromind,self.endind,1))
#     temp_end = int(-(len(ind_ori_0) % (self.dim_emb * self.num_eleinrow)))
#     ind_ori = ind_ori_0[:temp_end]
#     ind_mat_list = []
#     ind_pred_list = []
#     for idx in range(len(ind_ori)-2*self.dim_emb*self.num_eleinrow):
#         temp_mat = np.array(ind_ori[idx:idx+self.dim_emb*self.num_eleinrow]).reshape(-1,self.num_eleinrow).transpose()
#         temp_ind_pred_begin = temp_mat[-1, -1] + self.gap_topred
#         temp_ind_pred_end = temp_ind_pred_begin + self.num_pred + 1
#         ind_mat_list.append(temp_mat)
#         ind_pred_list.append(list(range(temp_ind_pred_begin, temp_ind_pred_end, 1)))
#     return ind_mat_list,ind_pred_list


#

if __name__ == '__main__':

    #dim_embs = 16
    # file_path = os.path.dirname(os.path.abspath(__file__))
    # mk_data_name = "mkdata_0730_morning.joblib"
    # file_full = os.path.join(file_path, mk_data_name)
    save_name  = genjname("ori")
    if False:
        outputsargs = Generate_UTS(MackeyGlass(),400,50).gettraindata()
        dump(outputsargs,save_name)

    else:
        outputsargs = load(save_name)
    trainset = outputsargs["train_input"]; train_mat=outputsargs["train_mat"]; targetset=outputsargs["train_target"]
    ind_mat_list = outputsargs["ind_mat_list"]; ind_pred_list = outputsargs["ind_pred_list"]
    lya_scale = outputsargs["lya_scale"]; timestamp = outputsargs["timestamp"]; dim_embs = outputsargs["dim_emb"]
    print(f"The lya_scale is {lya_scale}, the max_Lyapunov_exp is {MackeyGlass().maximum_lyapunov_estimated}.")

    fig_name_1 = genfname("in_out_examples")
    fig_name_2 = genfname("one_example")
    fig_name_3 = genfname("embvecs")

    plt.figure(1,figsize=(17, 8))
    for idxfig in list(range(0,9,1)):
        plt.subplot(3,3,idxfig+1)
        idxfig_ext = idxfig*ind_mat_list[0].size
        for idxline in list(range(0,train_mat[idxfig_ext].shape[0],1)):
            if idxline == 0:
                markersizes = 5
            else:
                markersizes = 2
            plt.plot(np.array(ind_mat_list[idxfig_ext])[idxline,:],np.array(train_mat[idxfig_ext])[idxline,:],'o',markersize=markersizes)
        plt.plot(ind_pred_list[idxfig_ext],targetset[idxfig_ext],'r*--')
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

    #plt.figure(3, figsize=(8, 8))
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
    #plt.figure()
    plt.show()
    #plt.ioff()
    plt.figure(2)
    #plt.figure()
    plt.show()
    #plt.ioff()
    plt.figure(3)

    #plt.figure(figsize=(8, 8))
    plt.show()
    #plt.ioff()


