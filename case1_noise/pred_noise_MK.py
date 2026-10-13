import matplotlib.pyplot as plt
import torch
from torch import optim
import torch.nn as nn
#from torch.utils.data import DataLoader, Dataset
import numpy as np
from joblib import dump, load

#from test_main_MK_100 import youwait
#from torchdeeponet import model as DONori
#import Fastformer
import os
from pathlib import Path
# from test_main_Lorenz_36 import Embedding as Embedding
# from test_main_Lorenz_36 import Unfolding as Unfolding
# from test_main_Lorenz_36 import FNN as FNN
# from test_main_Lorenz_36 import DeepONet as DeepONet
from case0_ori.main_Lorenz_36 import FastDeepONet as FastDeepONet
from case0_ori.main_Lorenz_36 import MyDataLoader as MyDataLoader
from pred_noise_Lorenz import LoadSplit_noise as LoadSplit_noise
from pred_noise_Lorenz import FigureDisplay as FigureDisplay
#from pred_noise_Lorenz import TrainFastDeepONet as TrainFastDeepONet
#from pred_noise_Lorenz import TestFastDeepONet as TestFastDeepONet
import torch.optim.lr_scheduler as lr_scheduler
import time
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


class TrainFastDeepONet:
    def __init__(self, X_Train, Y_Train, X_Test, Y_Test,
                 att_dim_emb, att_dim_delay,
                 layer_sizes_branch_l=128, layer_size_branch_r=128,
                 layer_sizes_trunk_l=128, layer_sizes_trunk_r=128,
                 activation=nn.functional.relu, bias=True, unstacked=True,
                 epochs=100000, batch_size=400,learning_rate=0.0002,flag_save_data_model=True,
                 save_j_name=None,save_f_names=None,):
        super().__init__()
        self.X_Train = X_Train
        self.Y_Train = Y_Train
        self.X_Test = X_Test
        self.Y_Test = Y_Test
        self.att_dim_emb = att_dim_emb
        self.att_dim_delay = att_dim_delay
        self.layer_sizes_branch_l = layer_sizes_branch_l
        self.layer_size_branch_r = layer_size_branch_r
        self.layer_sizes_trunk_l = layer_sizes_trunk_l
        self.layer_sizes_trunk_r = layer_sizes_trunk_r
        self.activation = activation
        self.bias = bias
        self.unstacked = unstacked
        self.epochs = epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.flag_save_data_model=flag_save_data_model
        self.save_j_name = save_j_name
        self.save_f_names = save_f_names


    def trainfdon(self):
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        data = MyDataLoader(X_train=self.X_Train, y_train=self.Y_Train, X_test=self.X_Test, y_test=self.Y_Test)
        mynet = FastDeepONet(self.att_dim_emb, self.att_dim_delay,
                             self.layer_sizes_branch_l, self.layer_size_branch_r, self.layer_sizes_trunk_l, self.layer_sizes_trunk_r,
                             self.activation).to(device)
        criterion = nn.MSELoss()
        optimizer = optim.Adam(mynet.parameters(), lr=self.learning_rate)
        epochs = self.epochs
        batch_size = self.batch_size
        scheduler_1 = lr_scheduler.LinearLR(optimizer, start_factor=1, end_factor=0.03, total_iters=int(epochs * 0.9))

        elapsed_time = 0
        for epoch in range(1, epochs + 1):
            mynet.train()  # Set the model to training mode
            start_time = time.time()
            for i in range(0, self.X_Train[0].shape[0], batch_size):
                #mynet.train()  # Set the model to training mode
                batch_x, batch_y = data.get_train_batch(batch_size)

                # Forward pass
                outputs = mynet(batch_x)
                loss = criterion(outputs, batch_y)

                # Backward pass and optimization
                optimizer.zero_grad()  # Zero the gradients
                loss.backward()  # Compute gradients
                optimizer.step()  # Update model weights
            scheduler_1.step()
            elapsed_time += time.time() - start_time
            if epoch % 100 == 0 or epoch == 1:
                # print(f'Epoch [{epoch}/{epochs}], Loss: {loss.item():.8f}, Test Loss: {test_loss.item():.8f}')

                mynet.eval()  # Set the model to evaluation mode
                with torch.no_grad():
                    test_x, test_y = data.get_test_set()  # ((N1, 100), (N2, 1)), (N1, N2)
                    test_outputs = mynet(test_x)  # (N1, N2)
                    test_loss = criterion(test_outputs, test_y)
                    # print(f'Test Loss: {test_loss.item():.8f}')
                print(f'Epoch [{epoch}/{epochs}], Loss: {loss.item():.8f}, Test Loss: {test_loss.item():.8f}')
        if self.flag_save_data_model:
            """file_path = os.path.dirname(os.path.abspath(__file__))

            file_name = "fastdon_0729_Lorenz+Noise.joblib"
            data_net_lorenz_noise_full = os.path.join(file_path, data_net_lorenz_noise_name)"""
           #save_name = genjname(self.specnote)
            dump([data,mynet,elapsed_time], self.save_j_name)

        # plot example
        plotflag = True
        if plotflag:
            all_len, gap = self.Y_Test.shape
            plt.figure(2, figsize=(12, 4))
            plt.plot(test_outputs[::gap, :].flatten().to('cpu'), label="pred", linestyle='--', color='red',linewidth=3)
            plt.plot(test_y[::gap, :].flatten().to('cpu'), label="ref", linestyle='-', color='blue',linewidth=2)
            plt.legend()

            #fig_description = 'Lorenz_lya=0.891710_' + f"TestLoss={test_loss.item():.8f}"+"_Noise"
            #save_name = genfname(self.specnote)
            #fig_full = os.path.join(file_path,fig_description)
            plt.savefig(self.save_f_names[0], dpi=300, bbox_inches='tight')
            plt.show()

            all_bias = np.array(test_outputs.to('cpu')) - np.array(test_y.to('cpu'))
            plt.figure(3, figsize=(4, 4))
            plt.hist(all_bias.flatten(), bins=80, color='b')
            plt.savefig(self.save_f_names[1], dpi=300, bbox_inches='tight')
            plt.show()

        return data, mynet,elapsed_time
from case1_noise.utils.metrics import metric
from case1_noise.utils.metrics import metric_std, metric_vec

import csv
class TestFastDeepONet:
    def __init__(self, mynet,test_x,test_y,save_f_name):
        self.mynet = mynet.to("cuda:0" if torch.cuda.is_available() else "cpu")
        self.test_x = (test_x[0].to("cuda:0" if torch.cuda.is_available() else "cpu"),
                       test_x[1].to("cuda:0" if torch.cuda.is_available() else "cpu"))
        self.test_y = test_y.to("cuda:0" if torch.cuda.is_available() else "cpu")
        self.save_f_name = save_f_name
    def testfdon(self):
        #device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        criterion = nn.MSELoss()
        self.mynet.eval()  # Set the model to evaluation mode
        with torch.no_grad():
            test_outputs = self.mynet(self.test_x)  # (N1, N2)
            test_loss = criterion(test_outputs, self.test_y)
            dtw = 'not calculated'
            mae, mse, rmse, mape, mspe, nmse, nrmse = metric(test_outputs.cpu().numpy().flatten(),
                                                             self.test_y.cpu().numpy().flatten())
            mae_std, mse_std, rmse_std, mape_std, mspe_std, nmse_std, nrmse_std = metric_std(test_outputs.cpu().numpy(),self.test_y.cpu().numpy())
            test_errors = test_outputs.cpu().numpy()-self.test_y.cpu().numpy()
            #mse_err_vec,mse_err_std = metric_vec(test_outputs.cpu().numpy(),self.test_y.cpu().numpy())
        """plt.figure(4,figsize=(12, 4))
        plt.errorbar(np.linspace(1,len(mse_err_vec),100),mse_err_vec,yerr=mse_err_std,fmt='o',label='MSE',ecolor='r',color='b',elinewidth=1, capsize=2,ms=3)
        plt.savefig(self.save_f_name, dpi=300, bbox_inches='tight')
        plt.show()"""
        fig,axes = plt.subplots(figsize=(12, 4))
        axes.plot(test_errors.T,color='steelblue',alpha=0.02,linewidth=0.5)
        time_steps = np.arange(test_errors.shape[1])
        median_pred = np.median(test_errors, axis=0)
        axes.plot(time_steps, median_pred, color='navy', linewidth=2, label='Median')
        perc_25 = np.percentile(test_errors, 25, axis=0)
        perc_75 = np.percentile(test_errors, 75, axis=0)
        axes.fill_between(time_steps, perc_25, perc_75, color='steelblue', alpha=0.3, label='25-75%')
        perc_5 = np.percentile(test_errors, 5, axis=0)
        perc_95 = np.percentile(test_errors, 95, axis=0)
        axes.fill_between(time_steps, perc_5, perc_95, color='steelblue', alpha=0.1, label='5-95%')
        axes.set_xlabel('Timesteps')
        axes.set_ylabel('Error')
        axes.legend()
        plt.savefig(self.save_f_name, dpi=300, bbox_inches='tight')
        plt.show()
        return test_loss.item(), mae, mse, rmse, mape, mspe,nmse,nrmse,mae_std, mse_std, rmse_std, mape_std, mspe_std, nmse_std, nrmse_std




if __name__ == '__main__':
    import random
    noise_strength = 0.20
    fix_seed = 2025
    random.seed(fix_seed)
    torch.manual_seed(fix_seed)
    np.random.seed(fix_seed)
    specnote = str(noise_strength)
    save_j_name = genjname(specnote)
    save_f_name_1 = genfname(specnote)
    save_f_name_2 = genfname(specnote+"_error")
    # save_f_name_1 = genfname(specnote+"example")
    # save_f_name_2 = genfname(specnote+"test")
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    X_Train, Y_Train, X_Test, Y_Test, Dim_emb, Dim_delay = LoadSplit_noise("F:\\PythonProject_DON\\case1_noise\\testgendata_MK_noise_0.2.joblib", 0.8).ldslt()
    #FigureDisplay(X_Train, Y_Train, X_Test, Y_Test, Dim_emb, Dim_delay,save_f_name_1).dodisplay()
    flag_train = False
    mg_epochs = 1000; mg_lr = 0.001; mg_batch_size = 400
    # save_name = genjname(specnote)
    if flag_train:
        data,mynet,elapsed_time = TrainFastDeepONet(X_Train, Y_Train, X_Test, Y_Test,Dim_emb,Dim_delay,
                                       epochs=1000,batch_size = 400,
                                       save_j_name = save_j_name,save_f_names=(save_f_name_1,save_f_name_2)).trainfdon()
    else:
        data,mynet,elapsed_time = load(save_j_name)
    test_loss,mae, mse, rmse, mape, mspe,nmse,nrmse, mae_std, mse_std, rmse_std, mape_std, mspe_std, nmse_std, nrmse_std = TestFastDeepONet(mynet,X_Test,Y_Test,save_f_name_2[0:-4]+'_bar.pdf').testfdon()

    modelname = "MG"
    with open('train_record_new.csv', 'r') as file:
        reader = csv.reader(file)
        rows = list(reader)
    new_row = [modelname,elapsed_time,mg_epochs,mg_batch_size,mg_lr,test_loss,
               str(mae)+'±'+str(mae_std),str(mse)+'±'+str(mse_std),str(rmse)+'±'+str(rmse_std),str(mape)+'±'+str(mape_std),
               str(mspe)+'±'+str(mspe_std),str(nmse)+'±'+str(nmse_std),str(nrmse)+'±'+str(nrmse_std)]
    rows[22] = new_row
    with open('train_record_new.csv', "w", newline="") as file:
        writer = csv.writer(file)
        writer.writerows(rows)

    #youwait = 1 '±'