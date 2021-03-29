import seaborn as sns
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
from scipy.stats import norm
from matplotlib import colors
import manufacturing as manufacturing


data=pd.read_csv('289.csv')

x1 = pd.Series(data['ER'].values.T, name="ER 消光比")
x2 = pd.Series(data['TxPower'].values.T, name="Tx Power")
#x1=x1[x1>16.5]

usl=20
lsl=12
sigma = 3

manufacturing.ppk_plot(x1, lower_control_limit=lsl, upper_control_limit=usl)
#manufacturing.ppk_plot(x2, lower_control_limit=-7, upper_control_limit=-3)
#manufacturing.control_plot(x2, lower_control_limit=-7, upper_control_limit=-3)

sns.set()

#mu=x1.mean()
# stdev=x1.std() #为什么区别于 mu, std = norm.fit(x1) ???
mu, stdev = norm.fit(x1)
cpu = round((usl - mu) / (sigma * stdev),2)
cpl = round((mu - lsl) / (sigma * stdev),2)
#print(mu,stdev)
cp=(cpu+cpl)/2
cpk = min(cpu, cpl)
#print(cpu,cpl,cp,cpk)

lower_percent = 10000 * 100.0 * stats.norm.cdf(lsl, mu, stdev)
print(f'{lower_percent:.02f} PPM < lsl')

higher_percent = 10000 * 100.0 * (1 - stats.norm.cdf(usl, mu, stdev))
print(f'{higher_percent:.02f} PPM > usl')

# f, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2)
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

f, (ax1, ax2) = plt.subplots(1,2,sharey=False)
#ax1.set_ylim(0,200)

#sns.distplot(x1,bins=20, kde=True,ax=ax1)

# Plot the histogram.
plt.style.use('seaborn-white')
count, bins, ignored = ax1.hist(x1, bins=20, density=False, alpha=0.6, color='g')
ax2.hist(x2, bins=20, density=False, alpha=0.6, color='g')

#ax1.plot(bins, 1/(3 * np.sqrt(2 * np.pi)) * np.exp( - (bins - 2)**2 / (2 * 3**2) ), linewidth=2, color='r')

# https://matplotlib.org/stable/gallery/statistics/histogram_features.html#sphx-glr-gallery-statistics-histogram-features-py
# add a 'best fit' line
#y = (1 / (np.sqrt(2 * np.pi) * sigma)) * np.exp(-0.5 * ((bins - mu)/sigma)**2)

ax21 = ax1.twinx()
ax21.set_ylim(0,1)
# 生成横轴数据平均分布
x = np.linspace(16 - sigma * stdev, 16 + sigma * stdev, 100)
# 计算正态分布曲线
y = np.exp(-(x - 16) ** 2 / (2 * stdev ** 2)) / (np.sqrt(2 * np.pi) * stdev)
ax21.plot(x, y, '--k')

#print('mu=',mu,'\nstdev=',stdev,'\ncount=',count, '\nbins=',bins, ignored)

# Plot the PDF.
xmin, xmax = ax1.get_xlim()
x = np.linspace(xmin-1, xmax+1, 100)
p = norm.pdf(x, mu, stdev)
ax21.plot(x, p, 'r',  alpha=0.5)
ax1.axvline(mu-sigma*stdev, color="r", linestyle="--")
ax1.axvline(mu+sigma*stdev, color="r", linestyle="--")
ax1.axvline(lsl, color="k", linestyle="--")
ax1.axvline(usl, color="k", linestyle="--")
# add a 'best fit' line
#y = ((1 / (np.sqrt(2 * np.pi) * sigma)) *
#     np.exp(-0.5 * (1 / sigma * (bins - mu))**2))
#ax1.plot(bins, y, '--')

ax22 = ax2.twinx()
ax22.set_ylim(0,1)
mu, stdev = norm.fit(x2)
xmin, xmax = ax2.get_xlim()
x = np.linspace(xmin-1, xmax+1, 100)
p = norm.pdf(x, mu, stdev)
ax22.plot(x, p, 'r', linewidth=2, alpha=0.5)


#  matplotlib.axes.Axes.hist() 方法的接口
#n, bins, patches = plt.hist(x=txpower.T, bins='auto',alpha=0.7, rwidth=0.85)
#plt.grid(axis='y', alpha=0.75)
#plt.xlabel('Value')
#plt.ylabel('Frequency')

#ax1.text(11, 0.70, "count " + str(x1.count()) + "\nmean  " + str(round(x1.mean(),2))+"\nstd   "+ str(round(x1.std(),2))+ "\nmin   "+ str(x1.min())+ "\nmax   "+ str(x1.max()))
left, right = ax21.get_xlim()
bottom, top = ax21.get_ylim()
props = dict(boxstyle='round', facecolor='white', alpha=0.9, edgecolor='grey')
strings=x1.describe().reset_index().to_string(header=None, index=None)+str(f'\n{lower_percent:.02f} PPM < lsl')+str(f'\n{higher_percent:.02f} PPM > usl')
ax21.text(left+0.05*(right-left), 0.6*top, strings,bbox=props)

left, right = ax22.get_xlim()
bottom, top = ax22.get_ylim()
props = dict(boxstyle='round', facecolor='white', alpha=0.9, edgecolor='grey')
strings=x2.describe().reset_index().to_string(header=None, index=None)+str(f'\n{lower_percent:.02f} PPM < lsl')+str(f'\n{higher_percent:.02f} PPM > usl')
ax22.text(left+0.05*(right-left), 0.6*top, strings,bbox=props)

plt.show()