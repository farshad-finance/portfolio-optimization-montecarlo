#Portfolio Optimization Using Monte Carlo Simulation
# Farshad Noravesh
# github.com/farshad-nsh
# github.com/farshad-finance

import pandas as pd
import numpy as np
import pandas_datareader.data as web
import matplotlib.pyplot as plt
#Fetch data from yahoo and save under DataFrame named 'data'
# Farshad likes Facebook, Tesla, Google, Amazon
stock = ['GOOG', 'FB', 'TSLA', 'AMZN']
#data = web.DataReader(stock,data_source="yahoo",start='12/01/2017',end='12/31/2019')['Adj Close']
data = web.DataReader(stock,data_source="yahoo",start='12/01/2017',end='09/01/2020')['Adj Close']

#Arrange the data in ascending order
data=data.iloc[::1]
print(data.round(2))


#Compute stock returns and print the returns in percentage format
stock_ret = data.pct_change()
print (stock_ret.round(4)*100)


#Calculate mean returns and covariances of all four the stocks
mean_returns = stock_ret.mean()
cov_matrix = stock_ret.cov()
print('mean-returns:')
print (mean_returns)
print('cov_matrix:')
print (cov_matrix)



#Set the number of iterations to 10000 and define an array to hold the simulation results; initially set to all zeros
num_iterations = 20000
simulation_res = np.zeros((4+len(stock)-1,num_iterations))



for i in range(num_iterations):
    #Select random weights and normalize to set the sum to 1
    weights = np.array(np.random.random(4))
    weights /= np.sum(weights)
    #Calculate the return and standard deviation for every step
    portfolio_return = np.sum(mean_returns * weights)
    portfolio_std_dev = np.sqrt(np.dot(weights.T,np.dot(cov_matrix, weights)))
    #Store all the results in a defined array
    simulation_res[0,i] = portfolio_return
    simulation_res[1,i] = portfolio_std_dev
    #Calculate Sharpe ratio and store it in the array
    simulation_res[2,i] = simulation_res[0,i] / simulation_res[1,i]
    #Save the weights in the array
    for j in range(len(weights)):
        simulation_res[j+3,i] = weights[j]



sim_frame = pd.DataFrame(simulation_res.T,columns=['ret','stdev','sharpe',stock[0],stock[1],stock[2],stock[3]])
print sim_frame.head (5)
print sim_frame.tail (5)



#Spot the position of the portfolio with highest Sharpe Ratio
max_sharpe = sim_frame.iloc[sim_frame['sharpe'].idxmax()]
#Spot the position of the portfolio with minimum Standard Deviation
min_std = sim_frame.iloc[sim_frame['stdev'].idxmin()]
print "The portfolio for max Sharpe Ratio:\n", max_sharpe
print "The portfolio for min risk:\n", min_std



#Create a scatter plot coloured by various Sharpe Ratios with standard deviation on the x-axis and returns on the y-axis
plt.scatter(sim_frame.stdev,sim_frame.ret,c=sim_frame.sharpe,cmap='RdYlBu')
plt.xlabel('Standard Deviation')
plt.ylabel('Returns')
plt.ylim(0.00075,0.003)
plt.xlim(0.015,0.03)
#Plot a red star to highlight position of the portfolio with highest Sharpe Ratio
plt.scatter(max_sharpe[1],max_sharpe[0],marker=(5,1,0),color='r',s=600)
#Plot a blue star to highlight position of the portfolio with minimum Variance
plt.scatter(min_std[1],min_std[0],marker=(5,1,0),color='b',s=600)
plt.show()



#------------------------------------------------------------------------------------------
