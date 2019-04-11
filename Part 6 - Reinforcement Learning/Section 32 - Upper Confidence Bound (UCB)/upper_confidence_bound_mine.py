# Upper Confidence Bound

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('Ads_CTR_Optimisation.csv')

#Implementation UCB
import math
N=10000
d=10
ads_selected = []
numbres_selections = [0]*d
sums_rewards = [0]*d
total_reward = 0

for n in range(0,N):
    ad = 0
    max_ub = 0
    for i in range(0,d):
        if (numbres_selections[i] > 0):
            average_reward = sums_rewards[i]/numbres_selections[i]
            delta_i = math.sqrt(3/2 * math.log(n+1)/numbres_selections[i])
            ub = average_reward + delta_i
        else :
            ub = 1e300
        if ub > max_ub :
            max_ub = ub
            ad = i
    ads_selected.append(ad)
    numbres_selections[ad] = numbres_selections[ad] + 1
    reward = dataset.values[n,ad]
    sums_rewards[ad] = sums_rewards[ad]+ reward
    total_reward = total_reward + reward
    
    
# Visualising the results
plt.hist(ads_selected)
plt.title('Histogram of ads selections')
plt.xlabel('Ads')
plt.ylabel('Number of times each ad was selected')
plt.show()


# Visualising the results
plt.plot(numbres_selections, range(0,10))
plt.title('Histogram of ads selections')
plt.xlabel('Ads')
plt.ylabel('Number of times each ad was selected')
plt.show()