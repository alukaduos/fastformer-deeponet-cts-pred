import csv

import matplotlib.pyplot as plt
import matplotlib
matplotlib.use( 'TkAgg')
import torch
from torch import optim
import torch.nn as nn
import random
import torch.random
from torch.utils.data import DataLoader, Dataset
import numpy as np
import os
from pathlib import Path
from joblib import dump, load
import Fastformer
from case1_noise.utils.metrics import metric
import torch.optim.lr_scheduler as lr_scheduler

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
lya_scale = 37, pred_horizon = 36, dim_emb = 12, dim_delay = 3, Lorenz, lya_max = 
epoch = 100000, batch_size = 400, 
onelayer_att, dim_input_x = pred_horizon, dim_output_y = 1, branch_size = (128,128), trunk_size = (128,128), 
"""
class LoadSplit:
    def __init__(self, filename, trainratio=0.98):
        self.filename = filename
        self.trainratio = trainratio
    def ldslt(self):
        outputsargs = load(self.filename)
        #allsamples = torch.tensor(outputsargs["train_input"],dtype=torch.float32)
        allsamples = torch.tensor(np.array(outputsargs["train_input"]),dtype=torch.float32)

        num_samples = len(allsamples)
        num_train = int(np.round(num_samples*self.trainratio))
        num_test = num_samples - num_train
       # alltargets = torch.tensor(outputsargs["train_target"],dtype=torch.float32)
        alltargets = torch.tensor(np.array(outputsargs["train_target"]),dtype=torch.float32)

        num_locations = len(allsamples[0])
        all_loc = torch.tensor(np.array(list(range(num_locations))),dtype=torch.float32).unsqueeze(1)
        dim_emb, dim_delay = outputsargs["dim_emb"], outputsargs["train_mat"][0].shape[0]
        flag_shuffle = True
        if flag_shuffle:
            perm = torch.randperm(allsamples[:num_train].shape[0])
            train_vec = allsamples[perm]; train_out = alltargets[perm]
        else:
            train_vec = allsamples[:num_train]; train_out = alltargets[:num_train]
        return (train_vec, all_loc),train_out, (allsamples[-num_test:],all_loc), alltargets[-num_test:], int(dim_emb), int(dim_delay)

class Embedding:
    def __init__(self, data,dim_emb, dim_delay):
        self.data = data
        self.dim_emb = dim_emb
        self.dim_delay = dim_delay
    def doemb(self):
        emb_mat = torch.reshape(self.data, (-1,self.dim_delay, self.dim_emb)).transpose(1, 2)
        return emb_mat

class Unfolding:
    def __init__(self, data):
        self.data = data
    def dounf(self):
        unfolded_vec = torch.flatten(self.data.transpose(1,2),1,-1)
        return unfolded_vec

class MyDataLoader:
    """
    X_train: A tuple of two PyTorch tensors. The first element has the shape (`N1`, `dim1`), and the second element has the shape (`N2`, `dim2`).
    y_train: A PyTorch tensor of shape (`N1`, `N2`).
    """

    def __init__(self, X_train, y_train, X_test, y_test, shuffle=False, drop_last=True):
        if X_train[0].shape[0] != y_train.shape[0] or X_train[1].shape[0] != y_train.shape[1]:
            raise ValueError(
                "The training dataset does not have the right format."
            )
        if X_test[0].shape[0] != y_test.shape[0] or X_test[1].shape[0] != y_test.shape[1]:
            raise ValueError(
                "The testing dataset does not have the right format."
            )

        self.train_x, self.train_y = X_train, y_train
        self.test_x, self.test_y = X_test, y_test

        self.dataset_size = X_train[0].shape[0]
        self.shuffle = shuffle
        self.indices = np.arange(self.dataset_size)
        self.current_idx = 0
        self.drop_last = drop_last
        if shuffle:
            np.random.shuffle(self.indices)

    def get_train_batch(self, batch_size=None):
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        if batch_size is None:
            return self.train_x, self.train_y

        start_idx = self.current_idx
        end_idx = min(start_idx + batch_size, self.dataset_size)
        batch_indices = self.indices[start_idx:end_idx]
        self.current_idx = end_idx

        if self.current_idx >= self.dataset_size:
            self.current_idx = 0

        return (self.train_x[0][batch_indices].to(device), self.train_x[1].to(device)), self.train_y[batch_indices].to(device)

    def get_test_set(self):
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        return (self.test_x[0].to(device),self.test_x[1].to(device)), self.test_y.to(device)
# wheel 1
class FNN(nn.Module):
    def __init__(self, layer_sizes, activation):
        super().__init__()
        self.activation = activation

        self.linears = torch.nn.ModuleList()
        for i in range(1, len(layer_sizes)):
            self.linears.append(
                torch.nn.Linear(
                    layer_sizes[i - 1], layer_sizes[i]
                )
            )

    def forward(self, inputs):
        x = inputs
        for j, linear in enumerate(self.linears[:-1]):
            x = self.activation(linear(x))
        x = self.linears[-1](x)
        return x
# wheel 2
class DeepONet(nn.Module):
    """
    layer_sizes_branch: A list of integers with layer widths for the branch net.
    layer_sizes_trunk (list): A list of integers with layer widths for the trunk net.
    The last layer width of branch and trunk net have to be the same.
    activation: (callable): An elementwise activation function.
    bias: (bool): whether to use a bias after merging or not.
    unstacked: (bool): whether to use a single branch net or one for each output dimension.
    """

    def __init__(self, layer_sizes_branch, layer_sizes_trunk, activation=None, bias=True, unstacked=True):
        super().__init__()
        self.activation = activation
        self.bias = bias
        self.unstacked = unstacked  # if unstacked is false, stacked is true

        if layer_sizes_branch[-1] != layer_sizes_trunk[-1]:
            raise ValueError("output dimension of branch net and trunk net must match!")

        self.p = layer_sizes_branch[-1]

        if unstacked:
            self.branch = FNN(layer_sizes_branch, self.activation)
        else:
            layer_sizes_branch = layer_sizes_branch[:-1] + [1]
            self.branches = []
            for _ in range(self.p):
                self.branches.append(FNN(layer_sizes_branch, self.activation))
        self.trunk = FNN(layer_sizes_trunk, self.activation)
        self.b = torch.nn.Parameter(torch.tensor(0.0)) if bias else None

    def merge_branch_trunk(self, x_func, x_loc):
        y = torch.einsum("bi,ni->bn", x_func, x_loc)
        if self.bias:
            y += self.b
        return y

    def forward(self, inputs):
        x_func = inputs[0]
        x_loc = inputs[1]

        # forward the DeepONet
        if self.unstacked:
            branch_out = self.branch(x_func)
        else:
            branch_outs = []
            for i in range(self.p):
                branch_outs.append(self.branches[i](x_func))
            branch_out = torch.cat(branch_outs, dim=1)

        trunk_out = self.activation(self.trunk(x_loc))
        x = self.merge_branch_trunk(branch_out, trunk_out)

        return x

class FastDeepONet(nn.Module):
    def __init__(self, att_dim_emb, att_dim_delay, layer_sizes_branch_l, layer_size_branch_r, layer_sizes_trunk_l,
                 layer_sizes_trunk_r, activation, bias=True, unstacked=True):
        super().__init__()
        self.att_dim_emb = att_dim_emb
        self.att_dim_delay = att_dim_delay
        self.layer_sizes_branch_l = layer_sizes_branch_l
        self.layer_size_branch_r = layer_size_branch_r
        self.layer_sizes_trunk_l = layer_sizes_trunk_l
        self.layer_sizes_trunk_r = layer_sizes_trunk_r
        self.activation = activation
        self.bias = bias
        self.unstacked = unstacked
        self.fastlayer =  Fastformer.Fastformer(dim = self.att_dim_emb, decode_dim=self.att_dim_emb)
        self.donlayer = DeepONet(layer_sizes_branch =[att_dim_emb*att_dim_delay, self.layer_sizes_branch_l, self.layer_size_branch_r],
                                 layer_sizes_trunk = [1,self.layer_sizes_trunk_l, self.layer_sizes_trunk_r],
                                 activation= nn.functional.relu)
    def forward(self, inputs):
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        x_ori = inputs[0]
        x_loc = inputs[1]
        x_emb_o = Embedding(x_ori, self.att_dim_delay, self.att_dim_emb).doemb().to(torch.float32)
        mask = torch.ones(1, self.att_dim_emb).bool().to(device)
        x_att = self.fastlayer(x_emb_o, mask)
        #x_att = Fastformer.Fastformer(dim = self.att_dim_emb, decode_dim=self.att_dim_emb).forward(x_emb_o, mask)
        dx_don_in_vec = Unfolding(x_att.squeeze()).dounf()
        x_don_in_vec = dx_don_in_vec+x_ori
        #x_don_in_loc = torch.tensor(list(range(x_don_in_vec.shape[0])),dtype=torch.float32)
        #x_don_in_loc = torch.tensor(x_loc,dtype=torch.float32)
        x_don_in_loc = x_loc.detach().to(torch.float32).clone().requires_grad_(False)
        x_don_in_tuple = (x_don_in_vec, x_don_in_loc)
        #dim_don_in = x_don_in_loc.size(0)
        don_out_vec = self.donlayer(x_don_in_tuple)
        return don_out_vec




""" don_out_vec = DeepONet(
                [dim_don_in, self.layer_sizes_branch_l, self.layer_size_branch_r],
                [1,self.layer_sizes_trunk_l, self.layer_sizes_trunk_r],
                self.activation,
                self.bias,
                self.unstacked
            ).forward(x_don_in_tuple)
"""
if __name__ == "__main__":
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    fix_seed = 2025
    #random.seed(fix_seed)
    torch.random.manual_seed(fix_seed)
    torch.manual_seed(fix_seed)
    np.random.seed(fix_seed)
    X_Train, Y_Train, X_Test, Y_Test, Dim_emb, Dim_delay = LoadSplit("gendata_Lorenz_ori.joblib",0.9).ldslt()
    """testre = FastDeepONet(Dim_emb, Dim_delay,
                          16,32,16,32,
                          nn.functional.relu).forward(X_Train[0])"""
    data = MyDataLoader(
        X_train=X_Train, y_train=Y_Train, X_test=X_Test, y_test=Y_Test
    )
    mynet = FastDeepONet(Dim_emb, Dim_delay,
                          128,128,128,128,
                          nn.functional.relu).to(device)
    criterion = nn.MSELoss()
    inital_lr = 0.001
    optimizer = optim.Adam(mynet.parameters(), lr=inital_lr)
    epochs = 1000
    batch_size = 128
    #scheduler_1 = lr_scheduler.MultiStepLR(optimizer, milestones=list(range(0, epochs, 130)), gamma=0.95)
    scheduler_1 = lr_scheduler.LinearLR(optimizer, start_factor=1, end_factor=0.05,total_iters=int(epochs*0.99))
    #scheduler_2 = lr_scheduler.CyclicLR(optimizer, base_lr=0.0001, max_lr=0.001, step_size_up=10, step_size_down=20)
    save_name = genjname('ori')
    flag_train = False
    import time
    if flag_train:
        elapsed_time = 0
        for epoch in range(1, epochs + 1):
            start_time = time.time()
            #mynet.train()  # Set the model to training mode

            for i in range(0, X_Train[0].shape[0], batch_size):
                mynet.train()  # Set the model to training mode
                batch_x, batch_y = data.get_train_batch(batch_size)

                # Forward pass
                outputs = mynet(batch_x)
                loss = criterion(outputs, batch_y)

                # Backward pass and optimization
                optimizer.zero_grad()  # Zero the gradients
                loss.backward()  # Compute gradients
                optimizer.step()  # Update model weights
            elapsed_time += time.time() - start_time
            #scheduler_2.step()
            scheduler_1.step()
            if epoch % 50 == 0 or epoch == 1:
                #print(f'Epoch [{epoch}/{epochs}], Loss: {loss.item():.8f}, Test Loss: {test_loss.item():.8f}')

                mynet.eval()  # Set the model to evaluation mode
                with torch.no_grad():
                    test_x, test_y = data.get_test_set()  # ((N1, 100), (N2, 1)), (N1, N2)
                    test_outputs = mynet(test_x)  # (N1, N2)
                    test_loss = criterion(test_outputs, test_y)
                    #print(f'Test Loss: {test_loss.item():.8f}')
                print(f'Epoch [{epoch}/{epochs}], Loss: {loss.item():.8f}, Test Loss: {test_loss.item():.8f}')
                print(f'Learning rate: {optimizer.param_groups[0]["lr"]:.8f}')

            # plot example
        dump(mynet, save_name)
    else:
        mynet = load(save_name)
        mynet.eval()  # Set the model to evaluation mode
        with torch.no_grad():
            test_x, test_y = data.get_test_set()  # ((N1, 100), (N2, 1)), (N1, N2)
            test_outputs = mynet(test_x)  # (N1, N2)
            test_loss = criterion(test_outputs, test_y)
            test_errors = test_outputs.cpu().numpy() - test_y.cpu().numpy()
        fig, axes = plt.subplots(figsize=(12, 4))
        err_back_color = 'steelblue'
        fig_name = genfname("ori")
        axes.plot(test_errors.T, color=err_back_color, alpha=0.02, linewidth=0.5)
        time_steps = np.arange(test_errors.shape[1])
        median_pred = np.median(test_errors, axis=0)
        axes.plot(time_steps, median_pred, color='navy', linewidth=2, label='Median')
        perc_25 = np.percentile(test_errors, 25, axis=0)
        perc_75 = np.percentile(test_errors, 75, axis=0)
        axes.fill_between(time_steps, perc_25, perc_75, color=err_back_color, alpha=0.3, label='25-75%')
        perc_5 = np.percentile(test_errors, 5, axis=0)
        perc_95 = np.percentile(test_errors, 95, axis=0)
        axes.fill_between(time_steps, perc_5, perc_95, color=err_back_color, alpha=0.1, label='5-95%')
        axes.set_ylim([-10, 10])
        axes.set_xlabel('Timesteps')
        axes.set_ylabel('Error')
        axes.legend()
        plt.savefig(fig_name[0:-4]+'_bar.pdf', dpi=300, bbox_inches='tight')
        plt.show()
    csvname = "test_record_Lorenz_ori.csv";model_name = 'FastDON'
    #print(......)
    step_horizons = list(range(2,36,3))
    for hidx,hval in enumerate(step_horizons):
        mae, mse, rmse, mape, mspe, nmse, nrmse = metric(np.array(test_outputs.to('cpu'))[:,hval],np.array(test_y.to('cpu'))[:,hval])
        with open(csvname, "r") as f:
            reader = csv.reader(f)
            rows = list(reader)
        new_rows = [model_name,'ela_time',hval,epochs,batch_size,inital_lr,np.array(test_loss.to('cpu')),mae,mse,rmse,mape,nmse,nrmse]
        rows[hidx+1] = new_rows
        with open(csvname, "w",newline="") as f:
            writer = csv.writer(f)
            writer.writerows(rows)


    all_len,gap = Y_Test.shape
    plt.figure(2,figsize=(12,4))
    plt.plot(test_outputs[::gap, :].flatten().to('cpu'), label="pred",linestyle='--', color='red',linewidth=3)
    plt.plot(test_y[::gap, :].flatten().to('cpu'), label="ref",linestyle='-', color='blue',linewidth=2)
    plt.legend()
    plt.show()
    #fig_description = 'Lorenz_lya=0.891710_'+f"TestLoss={test_loss.item():.8f}"
    fig_name = genfname("ori")
    plt.savefig(fig_name,dpi=300,bbox_inches='tight')

    from case1_noise.utils.metrics import metric
    import csv

    csvname_2 = "test_record_MG_ori.csv";    model_name = 'FastDON'
    mae, mse, rmse, mape, mspe, nmse, nrmse = metric(np.array(test_outputs.to('cpu')),
                                                     np.array(test_y.to('cpu')))
    if flag_train:
        with open(csvname_2, "r") as f:
            reader = csv.reader(f)
            rows = list(reader)
        new_rows = [model_name, elapsed_time, epochs, batch_size, inital_lr, np.array(loss.item()), mae, mse,
                    rmse, mape, mspe, nmse, nrmse]
        rows[6] = new_rows
        with open(csvname_2, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerows(rows)
    fig_name_2 = genfname("error_ori")
    all_bias = np.array(test_outputs.to('cpu')) - np.array(test_y.to('cpu'))

    plt.figure(3, figsize=(4, 4))
    plt.xlim(-5, 5)
    plt.hist(all_bias.flatten(), bins=200, color='b',)
    plt.savefig(fig_name_2, dpi=300, bbox_inches='tight')
    plt.show()
    youwait = 1
"""
emb_out_vec = Embedding(X_Train[0][0], Dim_emb, Dim_delay).doemb().unsqueeze(0).to(torch.float32)
mask = torch.ones(1, Dim_emb).bool()
MyAtt =  Fastformer.Fastformer(dim = Dim_emb,decode_dim=Dim_emb)
att_vec = MyAtt(emb_out_vec,mask)
don_in_vec = Unfolding(att_vec.squeeze()).dounf()
doc_in_loc = torch.tensor(list(range(don_in_vec.shape[0])),dtype=torch.float32)
don_in_tuple = (don_in_vec.unsqueeze(0),doc_in_loc.unsqueeze(1))
dim_don_in = doc_in_loc.size(0)
MyDON = DeepONet(
    [dim_don_in,13 ,40],
    [1, 16,40],
    nn.functional.relu,
    bias=True,
    unstacked=True
)
don_out_vec = MyDON(don_in_tuple)
"""





