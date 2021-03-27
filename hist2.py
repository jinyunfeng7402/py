import seaborn as sns
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
from scipy.stats import norm
from matplotlib import colors

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

f, (ax1, ax2) = plt.subplots(1,2, sharey=True)

sns.histplot(x1,bins=20, kde=True,ax=ax1)
#
mu, std = norm.fit(x1)

# Plot the histogram.
count, bins, ignored = ax1.hist(x1, bins=20, density=True, alpha=0.6, color='g')
ax2.hist(x2, bins=20, density=True, alpha=0.6, color='g')
ax1.plot(bins, 1/(3 * np.sqrt(2 * np.pi)) * np.exp( - (bins - 2)**2 / (2 * 3**2) ), linewidth=2, color='r')
print(count, bins, ignored)
# Plot the PDF.
xmin, xmax = ax1.get_xlim()
x = np.linspace(xmin-1, xmax+1, 100)
p = norm.pdf(x, mu, std)
ax1.plot(x, p, 'r', linewidth=2)
ax1.axvline(x1.min(), color="k", linestyle="--")
ax1.axvline(x1.max(), color="k", linestyle="--")

mu, std = norm.fit(x2)
xmin, xmax = ax2.get_xlim()
x = np.linspace(xmin-1, xmax+1, 100)
p = norm.pdf(x, mu, std)
ax2.plot(x, p, 'r', linewidth=2)
ax2.axvline(usl, color="k", linestyle="--")
ax2.axvline(lsl, color="k", linestyle="--")

#  matplotlib.axes.Axes.hist() 方法的接口
#n, bins, patches = plt.hist(x=txpower.T, bins='auto',alpha=0.7, rwidth=0.85)
#plt.grid(axis='y', alpha=0.75)
#plt.xlabel('Value')
#plt.ylabel('Frequency')

#ax1.text(11, 0.70, "count " + str(x1.count()) + "\nmean  " + str(round(x1.mean(),2))+"\nstd   "+ str(round(x1.std(),2))+ "\nmin   "+ str(x1.min())+ "\nmax   "+ str(x1.max()))
ax1.text(12, 0.625, x1.describe().reset_index().to_string(header=None, index=None))
ax2.text(-6.5, 0.625, x2.describe().reset_index().to_string(header=None, index=None))
#maxfreq = n.max()
# 设置y轴的上限
#plt.ylim(ymax=np.ceil(maxfreq / 10) * 10 if maxfreq % 10 else maxfreq + 10)

plt.show()