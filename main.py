# implement monte-carlo sim to simulate a stock portfolio
import datetime as dt
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pandas_datareader import data as pdr

#import data
def get_data(stocks, start, end):
    stockData = pdr.get_data_yahoo(stocks, start, end)
    stockData = stockData['Close']
    returns = stockData.pct_change()
    meanReturns = returns.mean()
    coMatrix = returns.cov()
    return meanReturns, coMatrix

stockList = ['CBA', 'BHP', 'TLS', 'NAB', 'WBC', 'STO']
stocks = [stock + '.AX' for stock in stockList]
endDate = dt.datetime.now()
startDate = endDate - dt.timedelta(days=300)

meanReturns, coMatrix = get_data(stocks, startDate, endDate)

print(meanReturns)

weights = np.random.random(len(meanReturns)) # 0 to 1
weights /= np.sum(weights)

# monte carlo method
# num of sims
mc_sims = 100
time = 100 #days

meanMatrix = np.full(shape=(time, len(weights)), fill_value=meanReturns)
meanMatrix = meanMatrix.time

portf_sims = np.full(shape=(time, mc_sims), fill_value=0.0) #shape is factor of sims/time

init_portf = 10000

for m in range (0, mc_sims):
# mean is lower triangle plus find from cholesky decomposition -> rep covariance matrix
    z = np.random.normal(size=(time, len(weights)))
    l = np.linalg.cholesky(coMatrix)
    dailyRet = meanMatrix + np.inner(l,z)
    portf_sims[:m] = np.cumprod(np.inner(weights, dailyRet.time) + 1)*init_portf # cumulative effect of daily changes
    

plt.plot(portf_sims)
plt.ylabel('portfolio val')
plt.xlabel('days')
plt.title('monte-carlo simulation of a portfolio')
plt.show()

