from dysts.flows import MackeyGlass
from dysts.analysis import find_lyapunov_exponents as lyaexp
import matplotlib.pyplot as plt
import matplotlib
import os
import torch
import numpy as np
matplotlib.use( 'Agg')
from joblib import dump, load
from testgendata_Lorenz_noise import Generate_UTS_Noise as GeneUTSNois
from testgendata_Lorenz_noise import ResultDisplay as ResuDisp
import os
from pathlib import Path
#from case1_noise.testgendata_Lorenz_noise import noise_strength
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

if __name__ == '__main__':
    #dim_embs = 16
    # file_path = os.path.dirname(os.path.abspath(__file__))
    # file_name = "mkdata_noise.joblib"
    # file_full = os.path.join(file_path, file_name)
    noise_strength = 0.20
    print("Noise Strength is:", noise_strength)
    save_name = genjname(str(noise_strength))
    save_f_name = (genfname(str(noise_strength)+"in_out_examples"),
                   genfname(str(noise_strength)+"one_example"),
                   genfname(str(noise_strength)+"embvecs"))
    if True:
        outputsargs = GeneUTSNois(MackeyGlass(),400,50,noise_strength = noise_strength).gettraindata()
        dump(outputsargs,save_name)
    else:
        outputsargs = load(save_name)
    ResuDisp(outputsargs,sampleconti=True,row_figs=3,col_figs=3,save_f_names=save_f_name).dodisplay()
