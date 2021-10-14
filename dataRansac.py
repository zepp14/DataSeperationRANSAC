

import numpy as np
import os
import matplotlib.pyplot as plt
import scipy.stats




def splitData(X_array,Y_array, MAXITER = 450):
    #Loop through RANSAC algo
    counter = 0
    szData =  len(Y_array)
    #fuse data
    DataSet = np.vstack((X_array,Y_array)).transpose()
    #print(DataSet)
    BinSizes = []
    CoeffArray = []
    ThreshArray = []

    DataGroup1 = []
    DataGroup2 = []

    while counter <= MAXITER:
        try:
            #Select three unique points at random
            DataGroup1 = []
            DataGroup2 = []
            rng = np.random.default_rng()
            P_rand = rng.choice(DataSet, size=3 )
            #print(P_rand )
            #solve for parabola model using the 3 random points 
            # model -> y = a*x**2 + b*x + c
            # Given three points a system of linear equations is solved to find [a b c]
            x1 = P_rand[0,0]
            x2 = P_rand[1,0]
            x3 = P_rand[2,0]
            Y = np.matrix([P_rand[0,1],P_rand[1,1],P_rand[2,1]]).transpose()
            A = np.matrix([[x1**2, x1, 1.0],
                        [x2**2, x2, 1.0],
                        [x3**2, x3, 1.0]])

            #Coed -> = [a,b,c]
            Coef = np.linalg.inv(A) @ Y

            CostFuncVal = []
            for P in DataSet:
                
                Y_model = Coef[0,0]*P[0]**2 + Coef[1,0]*P[0] + Coef[2,0]
                J = (Y_model - P[1])**2
                CostFuncVal.append(J)

            #Find Suitable Threshold using Historgram edges
            hist, bin_edges = np.histogram(CostFuncVal , bins=2)
            Thresh = bin_edges[1]

            BinSizes.append([hist[0], hist[1]])
            ThreshArray.append([Thresh])
            CoeffArray.append([Coef[0,0], Coef[1,0], Coef[2,0]])
            

            #Sort Points Based on
            
        except Exception as e:
            #print(e)
            #print("Singular Matrix Encountered Continuing to Next Iter")
            counter -= 1
            
        counter += 1
    
    BinSizes = np.array(BinSizes)
    ThreshArray = np.array(ThreshArray)
    CoeffArray = np.array(CoeffArray)
    
    ##Find best parabola model based on which maximizes the population of outlier group
    # Once the best index of bin sizes is found the data is split up based on the cost function value
    # and threshold using the values from the best index
    IDX_MAX = np.argmax(BinSizes[:,1], axis=None, out=None)

    Coef = CoeffArray[IDX_MAX, :]
    for P in DataSet:
        Y_model = Coef[0]*P[0]**2 + Coef[1]*P[0] + Coef[2]
        J = (Y_model - P[1])**2
        if J <= ThreshArray[IDX_MAX]:
            DataGroup1.append(P)
        else:
            DataGroup2.append(P)
    
    DataGroup1 = np.array(DataGroup1)
    DataGroup2 = np.array(DataGroup2)
    

    ## Improve robustness with a first order derivative hueristic
    #This computes the derivative of the of data groups, if derivatives are too high, it will run RANSAC again
    diffData1 = [ (DataGroup1[i-1,1] - DataGroup1[i,1])/(DataGroup1[i-1,0] - DataGroup1[i,0])  for i in range(1,len(DataGroup1))]
    sig1 = np.std(diffData1)
    
    diffData2 = [ (DataGroup2[i-1,1] - DataGroup2[i,1])/(DataGroup2[i-1,0] - DataGroup2[i,0])  for i in range(1,len(DataGroup2))]
    sig2 = np.std(diffData2)
   
    magOfdiff1 = np.floor(np.log10(sig1))
    magOfdata1 = np.floor(np.log10(np.mean(DataGroup1)))

    magOfdiff2 = np.floor(np.log10(sig2))
    magOfdata2 = np.floor(np.log10(np.mean(DataGroup2)))

    if magOfdiff1 >= magOfdata1 or magOfdiff2 >= magOfdata2 :
        DataGroup1, DataGroup2 = splitData(X_array,Y_array)
    else:
        pass

    return DataGroup1, DataGroup2



## Example Use ##

pathData_1 = "".join([os.getenv("HOME"),"/DataSeperationRANSAC/w1.npy"])
data1 = np.load(pathData_1)

pathData_2 = "".join([os.getenv("HOME"),"/DataSeperationRANSAC/w3.npy"])
data2 = np.load(pathData_2)

xAxis_Path = "".join([os.getenv("HOME"),"/DataSeperationRANSAC/gmm.npy"])
xData = np.load(xAxis_Path )


DataGroup1, DataGroup2 = splitData(xData, data2)

DataGroup3, DataGroup4 = splitData(xData, data1)

fig, ax = plt.subplots()
fig1, ax1 = plt.subplots()

ax.plot(DataGroup1[:,0],DataGroup1[:,1])
ax.plot(DataGroup2[:,0],DataGroup2[:,1])

ax1.plot(DataGroup3[:,0],DataGroup3[:,1])
ax1.plot(DataGroup4[:,0],DataGroup4[:,1])

plt.show()
