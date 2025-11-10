"""
@authors: Dr. Kyle Ian Williamson, Postdoctoral Fellow, ORISE, Hosted at USACE ERDC-EL,
          Dr. Harley McAlexander, Research Chemist, USACE ERDC-EL.
@email:  kyle.i.williamson@usace.army.mil,
         

This script was created to analyze datasets generate from Design of Experiment (DoE)
studies. For this study we used a Central-Composite Box design, but this script 
can be changed to process other designs as well. The main part of this code uses
the MS function from the ISLP Python Library developed for the textbook:

Introduction to Statistical Learning wtih Python

Full Citation: Gareth James, Daniela Witten, Trevor Hastie, Robert Tibshirani, and Jonathan Taylor. 
Intro to Statistical Learning with Python. Springer. 2023. https://doi.org/10.1007/978-3-031-38747-0.

While this code does not replace commercial statistical analysis software, it produces
results that are key for interpreting the results of response surface method (RSM) DoEs. 
"""
#%%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.pyplot import subplots
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor as VIF
from statsmodels.stats.anova import anova_lm
from ISLP import load_data
from ISLP.models import (ModelSpec as MS, summarize, poly)
import numpy as np
from scipy import stats
from scipy.stats import f
import seaborn as sns
from scipy.interpolate import griddata
from matplotlib import cm
from matplotlib.ticker import LinearLocator

# %% Load in the relevant csv dataset.
Dataset = pd.read_csv(r'Path to CSV File')

#Note: You will need to change the specific column title in y to the dependent variable/response factor you want to analyze.
#Note: You will also need to change the independent variables in X to match your own. If you have more than two independent variables, you'll need to look at those different combinations as well. 
#Note: You can look at linear and cubic relationships as well (example commented below X). If you do this, remember to change the equations below in the makeRange definition
y = Dataset['Depenedent Variable/Response Factor of Interest']
X = MS([('Independent Variable 1', 'Independent Variable 2'), poly('Independent Variable 1', degree=2,raw=True), poly('Independent Variable 2', degree=2,raw=True)]).fit_transform(Dataset)
# X = MS([('Alkylation', 'Hydrophilicity'), poly('Alkylation', degree=3), poly('Hydrophilicity', degree=3)]).fit_transform(Dataset)

model = sm.OLS(y, X)
results = model.fit()

print(summarize(results))
print("R-Squared:", results.rsquared)
print("RSE:", np.sqrt(results.scale))
#print("Residuals", results.resid)
#print("Fitted Values:", results.fittedvalues)


# %%
#Note: Run the code cell above to arrive at the best guess of the equation to describe the fit to the actual data. Then change the equation values/build in the function and at the bottom.
def lack_of_fit_test(realY, realX, alpha=0.05):
        """
        Pupose of this definition is to calculate the lack of fit test for a multilinear regression model.

        Args:
            y_actual (array-like): Actual values of the response variable.
            y_predicted (array-like): Predicted values of the response variable from the model.
            x (array-like): Input data (independent variables).
            alpha (float, optional): Significance level. Defaults to 0.05.

        Returns:
            tuple: F-statistic, p-value, and a boolean indicating whGASer to reject the null hypothesis.
        """
  
        y_predicted = results.predict(realX)
    
        n = len(realY)
        p = realX.shape[1] + 1  # Number of parameters in the model (including intercept)

        unique_x, indices = np.unique(realX, axis=0, return_inverse=True)
        m = len(unique_x) # Number of unique combinations of x
    
        if m == n:
            return "Cannot calculate Lack of Fit: No replicates found."
    
        SSPE = 0
        for i in range(m):
            y_values = realY[indices == i]
            SSPE += np.sum((y_values - np.mean(y_values))**2)
    
        SSE = np.sum((realY - y_predicted)**2)
        SSLOF = SSE - SSPE
    
        dfLOF = m - p
        dfPE = n - m
    
        if dfLOF <= 0 or dfPE <= 0:
             return "Cannot calculate Lack of Fit: degrees of freedom are not positive."
    
        MSLOF = SSLOF / dfLOF
        MSPE = SSPE / dfPE
    
        F0 = MSLOF / MSPE
        p_value = 1 - f.cdf(F0, dfLOF, dfPE)
    
        reject_null = p_value < alpha

        print("SSLOF:", SSLOF)
        print("MSLOF:", MSLOF)
        print("MSPE:", MSPE)
    
        return F0, p_value, reject_null

### Perform the lack of fit test
F0, p_value, reject_null = lack_of_fit_test(y, X)
   
print("F-statistic:", F0)
print("P-value:", p_value)
    
if isinstance(reject_null, str):
    print(reject_null)
elif reject_null:
    print("Reject the null hypothesis: Significant lack of fit.")
else:
    print("Fail to reject the null hypothesis: No significant lack of fit.")

# %% Plotting the actual vs. predicted values as a visual check. Replace 'COMP' with the column name with the list of values naming each trial (Example: Trial_1, Trial_2, etc.).

x_line= Dataset['COMP']
y_predicted = results.predict(X)

fit_line=y_predicted
#print(fit_line)
plot = subplots(figsize=(8,8))[1]
line1=plot.scatter(x_line, y, color='b')
plot.set_xlabel('COMP')
plot.set_ylabel('Depenedent Variable/Response Factor of Interest')
line2=plot.scatter(x_line, y_predicted, color='r')
plot.legend([line1, line2], ['y_actual', 'y_predicted'])
#plot.set_xticklabels(plot.get_xticks(), rotation=45)
plt.show()

# %% Getting the coefficients and such
# Hardcoded coeffs for raw=False; NOT suggested unless you are testing on a small dataset
# b0, b1, b2, b3, b4, b5, b6, b7 = -39.4251, -7.9492, -7.7164, -8.1603, 49.4552, 7.7164, -8.1603, -49.4552

# Alternatively, grab coeffs directly from results
coeffs = results.params

# %% Utility functions
def makeRange(myData,col,delta=10.0,nstep=100.0):
    minv = myData[col].min()
    maxv = myData[col].max()
    adj = maxv/delta
    inc = minv/nstep
    myRange = np.arange(minv-adj,maxv+adj,inc)
        
    return myRange

# Uses coefficients as returned by results.params
# Hardcoded to # and order of terms in the fit/model. We provide blocks for CUBIC and QUADRATIC equations. 
# We leave it up to you make similar blocks for linear or higher order polynomial fits. 

#For CUBIC Equations
# def myF(i,j,coefficients):
    
#     yp  = coefficients.iloc[0]
#     yp += coefficients.iloc[1] * i*j
#     yp += coefficients.iloc[2] * i
#     yp += coefficients.iloc[3] * i**2
#     #yp += coefficients.iloc[4] * i**3
#     yp += coefficients.iloc[5] * j
#     yp += coefficients.iloc[6] * j**2
#     #yp += coefficients.iloc[7] * j**3
 
#     return yp

#For QUADRATIC Equations
def myF(i,j,coefficients):
    
    yp  = coefficients.iloc[0]
    yp += coefficients.iloc[1] * i*j
    yp += coefficients.iloc[2] * i
    yp += coefficients.iloc[3] * i**2
    yp += coefficients.iloc[4] * j
    yp += coefficients.iloc[5] * j**2
 
    return yp

# Assumes coefficients have been predefined elsewhere
# def myF2(i,j):
#     # intercept = np.ones(len(i))
#     # yp = b0*intercept
#     yp = b0
#     yp += b1*i*j
#     yp += b2*i
#     yp += b3*i**2
#     yp += b4*i**3
#     yp += b5*j
#     yp += b6*j**2
#     yp += b7*j**3
    
#     return yp


#%% Set up x,y ranges for contour plot
myA = Dataset['Independent Variable 1']
myH = Dataset['Independent Variable 2']

xmin = myA.min()
xmax = myA.max()
ymin = myH.min()
ymax = myH.max()

#%% Set up mesh grid for contour plot. The plus and minus values should reflect the range of the independent variables in your dataset. 
# If you do not make changes to the plus and minus in myX an myY, the contours may not show valuable information because you are outside a relevant range. 
#myX = np.arange(xmin-0.1,xmax+0.1,0.05)
#myY = np.arange(ymin-0.1,ymax+0.1,0.05)
myX = np.arange(xmin-0.0001,xmax+0.0001,0.00005)
myY = np.arange(ymin-1,ymax+1,0.5)
xx, yy = np.meshgrid(myX,myY)
myZ = myF(xx,yy,coeffs)
# myZ = myF2(xx,yy)


#%% 2-D contour plot
fig, ax2 = plt.subplots()
cp = ax2.contour(xx,yy,myZ, cmap='jet')
ax2.clabel(cp)
ax2.set_title('2D Contour Plot')
ax2.set_xlabel('Independent Variable 1')
ax2.set_ylabel('Independent Variable 2')
fig.colorbar(cp, label='Depenedent Variable/Response Factor of Interest', pad=0.2)
plt.savefig(r"Path to Output Folder\2D Response Surface Plot.tif", dpi=500)

#%% 3-D surface
fig3, ax3 = plt.subplots(subplot_kw={"projection": "3d"})
surf = ax3.plot_surface(xx,yy,myZ, cmap='jet')
ax3.clabel(surf)
ax3.set_title('3D Response Surface Plot')
ax3.set_xlabel('Independent Variable 1')
ax3.set_ylabel('Independent Variable 2')
fig.colorbar(surf, label= 'Depenedent Variable/Response Factor of Interest', pad=0.2)
plt.savefig(r"Path to Output Folder\3D Response Surface Plot.tif", dpi=500)
plt.show()

