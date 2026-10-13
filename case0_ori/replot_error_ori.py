# -*- coding: utf-8 -*-
"""
仅重画 main_Lorenz_36_error_ori.pdf, 把 x 轴限制到 [-5, 5]
复用 main_Lorenz_36.py 的加载+前向逻辑, 不重训模型
"""
import os, sys
import numpy as np
import torch
import torch.nn as nn
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# 导入 main_Lorenz_36 的类 (LoadSplit / MyDataLoader / FastDeepONet 等)
import main_Lorenz_36 as M
# 让 FastDeepONet 等类进入当前 __main__ 命名空间, 供 joblib/pickle 反序列化
from main_Lorenz_36 import *

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
M.device = device

fix_seed = 2025
torch.random.manual_seed(fix_seed)
torch.manual_seed(fix_seed)
np.random.seed(fix_seed)

# 与主脚本完全一致的数据划分
X_Train, Y_Train, X_Test, Y_Test, Dim_emb, Dim_delay = M.LoadSplit("gendata_Lorenz_ori.joblib", 0.9).ldslt()
data = M.MyDataLoader(X_train=X_Train, y_train=Y_Train, X_test=X_Test, y_test=Y_Test)

# 加载已训练模型 (与 genjname('ori') 同名)
save_name = M.genjname('ori')
print("加载模型:", save_name)
mynet = M.load(save_name)
mynet.eval()

with torch.no_grad():
    test_x, test_y = data.get_test_set()
    test_outputs = mynet(test_x)

all_bias = np.array(test_outputs.to('cpu')) - np.array(test_y.to('cpu'))

# 重画误差直方图, x 轴限制 [-5, 5]
fig_name_2 = M.genfname("error_ori")
print("输出图:", fig_name_2)
plt.figure(figsize=(4, 4))
plt.hist(all_bias.flatten(), bins=50, color='b')
plt.xlim(-5, 5)
plt.savefig(fig_name_2, dpi=300, bbox_inches='tight')
plt.close()
print("完成. 全部完成")
