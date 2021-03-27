import seaborn as sns
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
from scipy.stats import norm

sns.set()

data=pd.read_csv('289.csv')

x1 = pd.Series(data['ER'].values.T, name="ER 消光比")
x2 = pd.Series(data['TxPower'].values.T, name="Tx Power")
#x1=x1[x1<17]

usl=-3
lsl=-7
sigma = 3
u=x2.mean()
stdev=x2.std()
cpu = round((usl - u) / (sigma * stdev),2)
cpl = round((u - lsl) / (sigma * stdev),2)

cp=(cpu+cpl)/2
cpk = min(cpu, cpl)
print(cpu,cpl,cp,cpk)

# f, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2)
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

f, (ax1, ax2) = plt.subplots(1, 2 ,sharey=True)

sns.distplot(x1,bins=20,fit=norm, kde=True,ax=ax1)
sns.distplot(x2,bins=20,fit=norm, kde=True,ax=ax2)
#  matplotlib.axes.Axes.hist() 方法的接口
#n, bins, patches = plt.hist(x=txpower.T, bins='auto',alpha=0.7, rwidth=0.85)
#plt.grid(axis='y', alpha=0.75)
#plt.xlabel('Value')
#plt.ylabel('Frequency')

#ax1.text(11, 0.70, "count " + str(x1.count()) + "\nmean  " + str(round(x1.mean(),2))+"\nstd   "+ str(round(x1.std(),2))+ "\nmin   "+ str(x1.min())+ "\nmax   "+ str(x1.max()))
ax1.text(11, 0.625, x1.describe().reset_index().to_string(header=None, index=None))
ax2.text(-7.25, 0.625, x2.describe().reset_index().to_string(header=None, index=None))
#maxfreq = n.max()
# 设置y轴的上限
#plt.ylim(ymax=np.ceil(maxfreq / 10) * 10 if maxfreq % 10 else maxfreq + 10)

plt.show()