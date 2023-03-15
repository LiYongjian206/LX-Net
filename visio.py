import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import seaborn as sns
from  sklearn import manifold
from matplotlib import ticker

data = np.load('n.npy')

label = np.load('D:/RF_CNN/data/label_g5.npy')

def Z_ScoreNormalization(x,min,max_min):
    x = (x - min) / max_min;
    return x;

T = Z_ScoreNormalization(data, np.min(data), np.max(data)-np.min(data)) * 100

# T = np.exp(T)

print(np.mean(T))
print(np.std(T))
print(np.max(T))
print(np.min(T))
# y_ticks = range(0,2000,400)  # 自定义横纵轴标签
# x_ticks = range(0,4000,800)  # 自定义横纵轴标签

# plt.rcParams['font.sans-serif'] = ['SimHei']
ax = sns.heatmap(T[0:5000, 0:382], annot=False, cmap="viridis", vmax=100, vmin=0, xticklabels=False, yticklabels=False)

#  annot=True表示每一个格子显示数字;fmt='.0f'表示保留0位小数，同理fmt='.1f'表示保留一位小数
#  camp表示颜色
#  vmax=350, vmin=20表示右侧颜色条的最大最小值，在最大最小值外的颜色将直接以最大或最小值的颜色显示，
#  通过此设置就可以解决少数值过大从而使得大部分色块区别不明显的问题
#  xticklabels=x_ticks, yticklabels=y_ticks，横纵轴标签

# ax.set_xlabel('Eigenvalues')  # x轴标题
# ax.set_ylabel('Normal')  # y轴标题 Atrial Fibrillation、Normal
plt.xlim(0, 384)
plt.xticks([i for i in range(0, 384, 40)])
plt.ylim(0, 5000)
plt.yticks([i for i in range(0, 5000, 400)])
plt.xlabel("Eigenvalues")
plt.ylabel("Atrial Fibrillation")
# plt.title("heatmap")
plt.gca().xaxis.set_major_formatter(ticker.FormatStrFormatter('%1.0f'))
plt.gca().yaxis.set_major_formatter(ticker.FormatStrFormatter('%1.0f'))
plt.show()

figure = ax.get_figure()
figure.savefig('正常.jpg')  # 保存图片 房颤、正常

# data_tsne = manifold.TSNE(n_components=2, init='pca', random_state=50).fit_transform(data)
#
# '''嵌入空间可视化'''
# x_min, x_max = data_tsne.min(0), data_tsne.max(0)
# X_norm = (data_tsne - x_min) / (x_max - x_min)  # 归一化
# print(X_norm.shape)
# vis_x = X_norm[:,0]
# vis_y = X_norm[:,1]
# plt.figure(figsize=(4, 4))
# plt.scatter(vis_x[:5000], vis_y[0:5000], c='blue', label='$N$', marker='.', edgecolors='white')
# plt.legend()
# plt.scatter(vis_x[5000:10000], vis_y[5000:10000], c='red', label='$AF$', marker='*', edgecolors='white')
# plt.legend()
# plt.show()