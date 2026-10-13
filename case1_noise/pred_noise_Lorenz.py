import matplotlib.pyplot as plt
import torch
from torch import optim
import torch.optim.lr_scheduler as lr_scheduler
# from math import sqrt
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
import numpy as np
from joblib import dump, load
import time
#from PI_DeepONet2025.deepxde.deepxde.backend import interactive_install_paddle
from utils.metrics import metric
import random
import csv
#from test_main_MK_100 import youwait
#from torchdeeponet import model as DONori
# import Fastformer
import os
from pathlib import Path
#from test_main_Lorenz_36 import Embedding as Embedding
#from test_main_Lorenz_36 import Unfolding as Unfolding
#from test_main_Lorenz_36 import FNN as FNN
#from test_main_Lorenz_36 import DeepONet as DeepONet
from case0_ori.main_Lorenz_36 import FastDeepONet as FastDeepONet
from case0_ori.main_Lorenz_36 import MyDataLoader as MyDataLoader
class LoadSplit_noise:
    """outputs are on gpu"""
    def __init__(self, filename, trainratio=0.80):
        self.filename = filename
        self.trainratio = trainratio
    def ldslt(self):
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        outputsargs = load(self.filename)
        allsamples = torch.tensor(np.array(outputsargs["train_input_noise"]),dtype=torch.float32)#with noise
        num_samples = len(allsamples)
        num_train = int(np.round(num_samples*self.trainratio))
        num_test = num_samples - num_train
        alltargets = torch.tensor(np.array(outputsargs["train_target_noise"]),dtype=torch.float32)#with noise
        targets_ori = torch.tensor(np.array(outputsargs["train_target"]),dtype=torch.float32)#without noise
        num_locations = len(allsamples[0])
        all_loc = torch.tensor(list(range(num_locations)),dtype=torch.float32).unsqueeze(1)
        dim_emb, dim_delay = outputsargs["dim_emb"], outputsargs["train_mat"][0].shape[0]
        return (allsamples[:num_train].to(device), all_loc.to(device)),alltargets[:num_train].to(device), (allsamples[-num_test:].to(device),all_loc.to(device)), targets_ori[-num_test:].to(device), int(dim_emb), int(dim_delay)

class FigureDisplay:
    def __init__(self,X_Train, Y_Train, X_Test, Y_Test, Dim_emb, Dim_delay, save_f_name = None):
        self.X_Train = X_Train
        self.Y_Train = Y_Train
        self.X_Test = X_Test
        self.Y_Test = Y_Test
        self.Dim_emb = Dim_emb
        self.Dim_delay = Dim_delay
        self.save_f_name = save_f_name
    def dodisplay(self):
        input_x_loc = list(range(self.Dim_emb * self.Dim_delay))
        target_y_loc = list(range(self.Dim_emb * self.Dim_delay - 1, self.Dim_emb * self.Dim_delay * 2 - 1, 1))
        plt.figure(1)
        plt.subplot(211)
        plt.plot(input_x_loc, self.X_Train[0][0].to('cpu'), 'bo-', label='Train input with noise ')
        plt.plot(target_y_loc, self.Y_Train[0].to('cpu'), 'r*--', label='Train target with noise')
        plt.legend()
        plt.subplot(212)
        plt.plot(input_x_loc, self.X_Test[0][40].to('cpu'), 'go-', label='Test input with noise')
        plt.plot(target_y_loc, self.Y_Test[40].to('cpu'), 'r*--', label='Test target without noise')
        plt.legend()
        plt.show()
        plt.savefig(self.save_f_name, dpi=300, bbox_inches='tight')

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
                 epochs=1000, batch_size=32,learning_rate=0.0001,flag_save_data_model=True,
                 save_j_name = None,save_f_name=None,save_f_names=None):
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
        self.save_f_name = save_f_name
        self.save_f_names = save_f_names


    def trainfdon(self):
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        data = MyDataLoader(X_train=self.X_Train, y_train=self.Y_Train, X_test=self.X_Test, y_test=self.Y_Test,shuffle=True)
        mynet = FastDeepONet(self.att_dim_emb, self.att_dim_delay,
                             self.layer_sizes_branch_l, self.layer_size_branch_r, self.layer_sizes_trunk_l, self.layer_sizes_trunk_r,
                             self.activation).to(device)
        criterion = nn.MSELoss()
        optimizer = optim.Adam(mynet.parameters(), lr=self.learning_rate)
        epochs = self.epochs
        batch_size = self.batch_size
        scheduler_0 = lr_scheduler.MultiStepLR(optimizer, milestones=list(range(0,epochs,80)), gamma=0.8)
        scheduler_1 = lr_scheduler.LinearLR(optimizer, start_factor=1, end_factor=0.05, total_iters=int(epochs * 0.95))

        scheduler_2 = lr_scheduler.CosineAnnealingWarmRestarts(optimizer,T_0=100, T_mult=2, eta_min=1e-7, last_epoch=-1)



        for epoch in range(1, epochs + 1):
            mynet.train()  # Set the model to training mode

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
            #scheduler_2.step()



            if epoch % 10 == 0 or epoch == 1:
                # print(f'Epoch [{epoch}/{epochs}], Loss: {loss.item():.8f}, Test Loss: {test_loss.item():.8f}')

                mynet.eval()  # Set the model to evaluation mode
                with torch.no_grad():
                    test_x, test_y = data.get_test_set()  # ((N1, 100), (N2, 1)), (N1, N2)
                    test_outputs = mynet(test_x)  # (N1, N2)
                    test_loss = criterion(test_outputs, test_y)
                    # print(f'Test Loss: {test_loss.item():.8f}')
                print(f'Epoch [{epoch}/{epochs}], Loss: {loss.item():.8f}, Test Loss: {test_loss.item():.8f}, Learning rate: {optimizer.param_groups[0]["lr"]:.8f}')
                #print(f'Learning rate: {optimizer.param_groups[0]["lr"]:.8f}')


        if self.flag_save_data_model:
            """file_path = os.path.dirname(os.path.abspath(__file__))

            file_name = "fastdon_0729_Lorenz+Noise.joblib"
            data_net_lorenz_noise_full = os.path.join(file_path, data_net_lorenz_noise_name)"""
            # save_name = genjname(self.specnote)
            dump([data,mynet], self.save_j_name)

        # plot example
        plotflag = True
        if plotflag:
            all_len, gap = self.Y_Test.shape
            plt.figure(2,figsize=(12, 4))
            plt.plot(test_outputs[::gap, :].flatten().to('cpu'), label="pred", linestyle='--', color='red',linewidth=3)
            plt.plot(test_y[::gap, :].flatten().to('cpu'), label="ref", linestyle='-', color='blue',linewidth=2)
            plt.legend()
            plt.savefig(self.save_f_names[0], dpi=300, bbox_inches='tight')
            plt.show()
            #fig_description = 'Lorenz_lya=0.891710_' + f"TestLoss={test_loss.item():.8f}"+"_Noise"
            # save_name = genfname(self.specnote)
            #fig_full = os.path.join(file_path,fig_description)
            all_bias = np.array(test_outputs.to('cpu')) - np.array(test_y.to('cpu'))
            plt.figure(3, figsize=(4, 4))
            plt.xlim(-10,10)
            plt.hist(all_bias.flatten(), bins=200, color='b')
            plt.savefig(self.save_f_names[1], dpi=300, bbox_inches='tight')
            plt.show()


        return data, mynet
from case1_noise.utils.metrics import metric_std, metric_vec

class TestFastDeepONet:
    def __init__(self, mynet,test_x,test_y,save_f_name):
        self.mynet = mynet
        self.test_x = test_x
        self.test_y = test_y
        self.save_f_name = save_f_name
    def testfdon(self):
        criterion = nn.MSELoss()
        self.mynet.eval()  # Set the model to evaluation mode
        with torch.no_grad():
            test_outputs = self.mynet(self.test_x)  # (N1, N2)
            test_loss = criterion(test_outputs, self.test_y)

            dtw = 'not calculated'
            mae, mse, rmse, mape, mspe,nmse,nrmse = metric(test_outputs.cpu().numpy().flatten(), self.test_y.cpu().numpy().flatten())
            mae_std, mse_std, rmse_std, mape_std, mspe_std, nmse_std, nrmse_std = metric_std(test_outputs.cpu().numpy(),self.test_y.cpu().numpy())
            test_errors = test_outputs.cpu().numpy()-self.test_y.cpu().numpy()
            fig, axes = plt.subplots(figsize=(12, 4))
            axes.plot(test_errors.T, color='steelblue', alpha=0.02, linewidth=0.5)
            time_steps = np.arange(test_errors.shape[1])
            median_pred = np.median(test_errors, axis=0)
            axes.plot(time_steps, median_pred, color='navy', linewidth=2, label='Median')
            perc_25 = np.percentile(test_errors, 25, axis=0)
            perc_75 = np.percentile(test_errors, 75, axis=0)
            axes.fill_between(time_steps, perc_25, perc_75, color='steelblue', alpha=0.3, label='25-75%')
            perc_5 = np.percentile(test_errors, 5, axis=0)
            perc_95 = np.percentile(test_errors, 95, axis=0)
            axes.fill_between(time_steps, perc_5, perc_95, color='steelblue', alpha=0.1, label='5-95%')
            axes.set_ylim([-10,10])
            axes.set_xlabel('Timesteps')
            axes.set_ylabel('Error')
            axes.legend()
            plt.savefig(self.save_f_name, dpi=300, bbox_inches='tight')
            plt.show()
            # print('mse:{}, mae:{}, dtw:{}'.format(mse, mae, dtw))
            print('mse:{}, rmse:{}, mspe{}, mae:{}, mape:{}, dtw:{}'.format(mse, rmse, mspe, mae, mape, dtw))
            f = open("result_lorenz.txt", 'a')
            f.write('nothing' + "  \n")
            # f.write('mse:{}, mae:{}, dtw:{}'.format(mse, mae, dtw))
            f.write('mse:{}, rmse:{}, mspe{}, mae:{}, mape:{}, dtw:{}'.format(mse, rmse, mspe, mae, mape, dtw))
            f.write('\n')
            f.write('\n')
            f.close()

            # np.save(folder_path + 'metrics.npy', np.array([mae, mse, rmse, mape, mspe]))
            np.save( 'lorenz_metrics.npy', np.array([mse, rmse, mspe, mae, mape]))
            np.save( 'lorenz_pred.npy', test_outputs.cpu())
            np.save( 'lorenz_true.npy', self.test_y.cpu())
        return test_loss.item(), mae, mse, rmse, mape, mspe,nmse,nrmse

"""import importlib
dynmodule = importlib.import_module("case1_noise.testgendata_Lorenz_noise")
noise_stength = dynmodule.noise_strength"""

if __name__ == '__main__':
    noise_strength = 0.1
    fix_seed = 2025
    random.seed(fix_seed)
    torch.manual_seed(fix_seed)
    np.random.seed(fix_seed)
    save_j_name = genjname("data_mynet"+str(noise_strength))
    save_f_name_1= genfname(str(noise_strength)+"example")
    save_f_name_2= genfname(str(noise_strength)+"test")
    save_f_names = (genfname(str(noise_strength)),genfname(str(noise_strength)+"_error"))

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    X_Train, Y_Train, X_Test, Y_Test, Dim_emb, Dim_delay = LoadSplit_noise("testgendata_Lorenz_noise_0.1.joblib", 0.9).ldslt()
    #FigureDisplay(X_Train, Y_Train, X_Test, Y_Test, Dim_emb, Dim_delay,save_f_name_1).dodisplay()
    flag_train = False

    lor_epochs = 500; lor_batch_size = 128; init_lr = 0.001
    # specnote = str(noise_strength)
    # save_name = genjname(specnote)
    if flag_train:
        train_start_time = time.time()

        data,mynet = TrainFastDeepONet(X_Train, Y_Train, X_Test, Y_Test,
                                       Dim_emb,Dim_delay,epochs=lor_epochs, batch_size = lor_batch_size, learning_rate=init_lr,
                                       save_j_name = save_j_name,save_f_name=save_f_name_2,save_f_names=save_f_names).trainfdon()
        train_end_time = time.time()
        eap_time = train_end_time - train_start_time
        dump(eap_time,"eap_time.joblib")
    else:
        data,mynet = load(save_j_name)
        eap_time = load("eap_time.joblib")
    test_loss, mae, mse, rmse, mape, mspe,nmse,nrmse = TestFastDeepONet(mynet,X_Test,Y_Test,save_f_name_2[0:-4]+'_bar.pdf').testfdon()
    #the above test_loss = mse, so in the .csv, I change this to train_loss, manually input
    youwait = 1
    modelname = 'lorenz'
    with open('train_record.csv', 'r') as file:
        reader = csv.reader(file)
        rows = list(reader)
    new_row = [modelname,eap_time,lor_epochs,lor_batch_size,init_lr,test_loss,mae,mse,rmse,mape,mspe,nmse,nrmse]
    rows[2] = new_row
    with open('train_record.csv', "w", newline="") as file:
        writer = csv.writer(file)
        writer.writerows(rows)


