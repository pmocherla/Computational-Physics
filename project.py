# -*- coding: utf-8 -*-
"""
Created on Wed Nov 23 10:32:18 2016

@author: pm2014
"""
"""
This code is used to plot and minimise the probability density function of a set of data. The data analysed here
corresponds to the decay times of Do mesons, and the functions used to find the lifetime and fraction of signal 
parameters.

To use this code the following libraries are required:
    - Numpy
    - Scipy
    - Math
    - Matplotlib
    - Random
    
The code includes the following functions:
    - fitfunction
    - NLL and NLL_bg
    - function (1D and 2D)
    - minimiser
    - minimiser2d
    - minimiser2dquasi
    - minimiser2dnewton
    - abserror
    - abserror2D
    - stdev
    - contourplot
    
"""

import numpy as np
import matplotlib.pyplot as plt
import math as m
import random as rand
import scipy.integrate as integrate

#------------------------ Load data and parameters ---------------------------#
data = np.loadtxt("lifetime.txt")

#1D parabolic minimiser initial starting points
x0 = 0.2
x2 = 0.6

#Setting tolerance for minimisations and error calculations
tol = 0.00001

#Settings for 2d minimisation plotting
stepsize = 0.05 #set stepsize for tau and a plotting

gradient = True #set each minimisation method to True to calculate it
newton = False
quasi = False
error = False #Calculates uncertainty for each method if true
trials = 10 #set to number of trials required

#-------------------------- Fixed Parameters ----------------------------#
#DO NOT TOUCH
tau = np.arange(0.1,1.01,stepsize)
a = np.arange(0.05,1.01,stepsize)
t = data[:,0]
resol = data[:,1]
fitdata = np.zeros(t.size)
fitdatabg = np.zeros(t.size)
NLLdata = np.zeros(t.size)
NLLdatabg = np.zeros(t.size)
NLLsums = np.zeros(tau.size)
x = np.array([x0,(x0+x2)/2,x2])
xplot = []
yplot = []
h = 0.00001
alpha = 0.00001
y = np.zeros(x.size)
Z = np.zeros((a.size,tau.size))
bins = np.linspace(-2, 6, 100)
toler = [tol,tol]
aplot = []
tauplot = []
aplot1 = []
tauplot1 = []
aplot2 = []
tauplot2 = []
tautest = 0.1
sigmatest = 0.1
step = 0.0001

#------------------------ Analysis Functions -------------------------–#
"""NOT including background radiation"""

def fitfunction(time, lifetime, resolution):
    """ 
        returns the fit function of the dataset inputted.
        
        Args: 
        time: array eg. np.array([x1,x2,x3...])
        lifetime: float
        resolution: float                  
    """
    for i in range(0, time.size):
        fitdata[i] = (1/(2*lifetime))*np.exp((resolution**2/(2*lifetime**2))-(time[i]/lifetime))*m.erfc((1/np.sqrt(2))*((resolution/lifetime)-(time[i]/resolution)))
    return fitdata
    
def NLL(time, lifetime, error):
    """ 
        returns the negative log likelihood of the data inputted.
        
        Args: 
        time: array eg. np.array([x1,x2,x3...])
        lifetime: float
        error: array eg. np.array([x1,x2,x3...])                 
    """
    for i in range(0, time.size):
        NLLdata[i] = - np.log((1/(2*lifetime))*np.exp((error[i]**2/(2*lifetime**2))-(time[i]/lifetime))*m.erfc((1/np.sqrt(2))*((error[i]/lifetime)-(time[i]/error[i]))))
    return np.sum(NLLdata)
    
def function(x, data):
    """ 
        returns the 1D function (negative log likelihood) to be minimised.
        
        Args: 
        x: integer
        data: integer, number of datapoints to be included from the sample (default = set to 10000)   
    """
    return NLL(t[0:data], x, resol[0:data])
    
def minimiser(x, tol,data):
    """ 
        returns the 1D minimum coordinates of a dataset according to a defined function.
        
        Args: 
        x: 3D array eg. np.array([x1,x2,x3]), starting point of minimisation
        tolerace: float, tolerace of convergence
        data: integer, number of datapoints to be included from the dataset (default = set to 10000)           
    """
    for i in range(x.size):
        y[i] = function(x[i],data)
    x3 = 0.5*((x[2]**2-x[1]**2)*y[0]+(x[0]**2-x[2]**2)*y[1]+(x[1]**2-x[0]**2)*y[2])/((x[2]-x[1])*y[0]+(x[0]-x[2])*y[1]+(x[1]-x[0])*y[2])
    xplot.extend(x)
    yplot.extend(y)
    while (abs(np.amax(y) - np.amin(y)) > tol) and (function(x3, data) < np.amax(y)):
        x[y.argmax()] = x3
        y[y.argmax()] = function(x3,data)
        xplot.append(x3)
        yplot.append(function(x3,data))
        x3 = 0.5*((x[2]**2-x[1]**2)*y[0]+(x[0]**2-x[2]**2)*y[1]+(x[1]**2-x[0]**2)*y[2])/((x[2]-x[1])*y[0]+(x[0]-x[2])*y[1]+(x[1]-x[0])*y[2])
        continue
    else:
        xplot.append(x[y.argmin()])
        yplot.append(y[y.argmin()])
        return x[y.argmin()],y[y.argmin()]
        
def abserror(minimum, step,data):
    """ 
        returns the absolute error of the minimisation.
        
        Args: 
        minimum: 2D array eg. np.array([x1,x2]), minimised coordinates (can input the minimisation function directly)
        step: float, gradient calculation step length
        data: integer, number of datapoints to be included from the dataset (default= set to 10000)           
    """
    NLLplus = minimum[1] + 0.5
    startplus = minimum[0]
    while NLLplus > function(startplus,data):
        startplus = startplus + step
        continue
    else:
        return (startplus - minimum[0]) 
        
def stdev(data):
    """ 
        returns the standard deviation of the minimisation.
        
        Args: 
        data: integer, number of datapoints to be included from the dataset  (default = set to 10000)         
    """
    minimiser(x, tol,data)      
    d2P2 = (2/((x[1]-x[0])*(x[2]-x[0])*(x[2]-x[1])))*(y[0]*(x[2]-x[1])+y[1]*(x[0]-x[2])+y[2]*(x[1]-x[0]))
    stdev = 1/np.sqrt(d2P2)
    return stdev
    

"""With background radiation"""
def fitfunction_bg(time, lifetime, fraction, resolution):
    """ 
        returns the fit function of the dataset inputted.
        
        Args: 
        time: array eg. np.array([x1,x2,x3...])
        lifetime: float
        fraction: float
        resolution: float                  
    """
    for i in range(0,t.size):
        fitdata[i] = (fraction*(1/(2*lifetime))*np.exp((resolution[i]**2/(2*lifetime**2))-(time[i]/lifetime))*m.erfc((1/np.sqrt(2))*((resolution[i]/lifetime)-(time[i]/resolution)))) + np.exp((-time[i]**2)/(2*resolution[i]**2))*(1-fraction)/(resolution[i]*np.sqrt(2*np.pi))
    return fitdatabg

def NLL_bg(time, lifetime, fraction, resolution):
    """ 
        returns the negative log likelihood of the dataset inputted.
        
        Args: 
        time: array eg. np.array([x1,x2,x3...])
        lifetime: float
        fraction: float
        resolution: array eg. np.array([x1,x2,x3...])                
    """
    for i in range(0,t.size):
        NLLdatabg[i] = - np.log((fraction*(1/(2*lifetime))*np.exp((resolution[i]**2/(2*lifetime**2))-(time[i]/lifetime))*m.erfc((1/np.sqrt(2))*((resolution[i]/lifetime)-(time[i]/resolution[i])))) + np.exp((-time[i]**2)/(2*resolution[i]**2))*(1-fraction)/(resolution[i]*np.sqrt(2*np.pi)))
    return np.sum(NLLdatabg)
    
def function_bg(x,y):
    """ 
        returns the 2D function (negative log likelihood) to be minimised.
        
        Args: 
        x: integer
        y: integer 
    """
    return NLL_bg(t, x, y, resol)

def minimiser2d(start, step, alpha, tol):
    """ 
        returns the 2D minimum coordinates of a dataset using the gradient method, number of iterations.
        
        Args: 
        start: 2D array eg. np.array([x1,x2]), starting point of minimisation
        step: step size of gradient calculation
        alpha: gradient descent value
        tol: array eg. np.array([x1,x2]) tolerance of minimisation in each dimension        
    """
    tauplot.append(start[0])
    aplot.append(start[1])
    starting = function_bg(start[0], start[1])
    grad = np.zeros(2)
    grad[0] = (function_bg(start[0] + step,start[1]) - starting)/step
    grad[1] =  (function_bg(start[0],start[1]+ step) - starting)/step
    while np.all(abs((start - (start - alpha*grad))) > tol):
        start = start - alpha*grad
        tauplot.append(start[0])
        aplot.append(start[1])
        starting = function_bg(start[0], start[1])
        #print start
        grad[0] = (function_bg(start[0] + step,start[1]) - starting)/step
        grad[1] =  (function_bg(start[0],start[1]+ step) - starting)/step
        continue
    else:
        return start, len(tauplot)
      
def minimiser2dnewton(start,step,tol):
    """ 
        returns the 2D minimum coordinates of a dataset using the Newton method, number of iterations.
        
        Args: 
        start: 2D array eg. np.array([x1,x2]), starting point of minimisation
        step: step size of gradient calculation
        tol: array eg. np.array([x1,x2]) tolerance of minimisation in each dimension      
    """
    tauplot1.append(start[0])
    aplot1.append(start[1])
    H = np.identity((2))
    grad = np.zeros(2)
    starting =  function_bg(start[0], start[1])
    grad[0] = (function_bg(start[0] + step,start[1]) - starting)/step
    grad[1] =  (function_bg(start[0],start[1]+ step) - starting)/step
    while np.all((abs(start - (start - np.dot(np.linalg.inv(H),grad)))) > tol):
        H[0][0] = (function_bg(start[0]+2*step,start[1])-2*function_bg(start[0]+step,start[1])+starting)/step**2
        H[0][1] = (function_bg(start[0]+step,start[1]+step)-function_bg(start[0]+step,start[1])-function_bg(start[0],start[1]+ step)+starting)/step**2
        H[1][0] = (function_bg(start[0]+step,start[1]+step)-NLL_bg(t, start[0]+step,start[1],resol)-NLL_bg(t, start[0],start[1]+ step,resol)+starting)/step**2
        H[1][1] = (function_bg(start[0],start[1]+2*step)-2*function_bg(start[0],start[1]+step)+starting)/step**2
        start = (start - np.dot(np.linalg.inv(H),grad))
        tauplot1.append(start[0])
        aplot1.append(start[1])
        starting = function_bg(start[0], start[1])
        grad[0] = (function_bg(start[0] + step,start[1]) - starting)/step
        grad[1] = (function_bg(start[0],start[1]+ step) - starting)/step
        continue
    else:
        return start, len(tauplot1)
        
def minimiser2dquasi(start,step,alpha,tol):
    """ 
        returns the 2D minimum coordinates of a dataset using the Quasi-Newton method, number of iterations.
        
        Args: 
        start: 2D array eg. np.array([x1,x2]), starting point of minimisation
        step: step size of gradient calculation
        alpha: gradient descent value
        tol: array eg. np.array([x1,x2]) tolerance of minimisation in each dimension      
    """
    tauplot2.append(start[0])
    aplot2.append(start[1])
    G = np.identity((2))
    grad = np.zeros(2)
    gamma = np.zeros(2)
    gradplusone = np.zeros(2)
    starting = function_bg(start[0],start[1])
    grad[0] = (function_bg(start[0] + step,start[1]) - starting)/step
    grad[1] =  (function_bg(start[0],start[1]+ step) - starting)/step
    while np.all(abs(start -(start - alpha*np.dot(G,grad))) > tol):
        startplusone = start - alpha*np.dot(G,grad)
        tauplot2.append(start[0])
        aplot2.append(start[1])
        gradplusone[0] = (function_bg(startplusone[0] + step,startplusone[1]) - function_bg(startplusone[0],startplusone[1]))/step
        gradplusone[1] = (function_bg(startplusone[0],startplusone[1] + step) - function_bg(startplusone[0],startplusone[1]))/step
        delta = startplusone - start
        gamma = gradplusone - grad
        G = G + np.outer(delta,delta)/np.dot(gamma,delta)
        start = startplusone
        grad[0] = gradplusone[0]
        grad[1] = gradplusone[1]
        continue
    else:
        return start, len(tauplot2)

def contourplot():
    """
        returns the contour plot of the lifetime dataset, including the background signal
    """
    X, Y = np.meshgrid(tau, a)
    for i in range(a.size):
        print a.size-i
        for j in range(tau.size):
            Z[i][j] = NLL_bg(t,tau[j],a[i],resol)
    plt.contour(X, Y, Z, levels = np.arange(6000,24000,1000))
    clb = plt.colorbar()
    clb.ax.set_title('NLL')
    plt.title('Minimising Fraction of signal and Lifetime of sample')
    plt.xlabel('Lifetime/ ps', fontsize = 18)
    plt.ylabel('Fraction of signal', fontsize = 18)
    plt.ylim((0.05,1.0))
    plt.xlim((0.1,1.0))
    plt.show()
    
def abserror2(minimum,tol,step,tau):
    """ 
        returns the absolute error of the minimisation.
        
        Args: 
        minimum: array eg. np.array([x1,x2]), minimised coordinates (can input the minimisation function directly)
        tol: array eg. np.array([x1,x2]) tolerance of minimisation in each dimension
        step: float, gradient calculation step length
        tau: True or False, calculates tau error if True and fraction error if False           
    """
    if tau == True:
        NLLplus = NLL_bg(t,minimum[0], minimum[1],resol) + 0.5
        startplus = minimum[0]
        while NLLplus > NLL_bg(t, startplus,minimum[1], resol):
            startplus = startplus + step
            continue
        else:
           return (startplus - minimum[0])
    else:
        NLLplus = NLL_bg(t,minimum[0], minimum[1],resol) + 0.5
        startplus = minimum[1]
        while NLLplus > NLL_bg(t, minimum[0],startplus, resol):
            startplus = startplus + step
            continue
        else:
           return (startplus - minimum[1])       
         
def abserror2D(minimum,tol,step):    
    """ 
        returns the absolute error of the minimisation, for each dimension.
        
        Args: 
        minimum: array eg. np.array([x1,x2]), minimised coordinates (can input the minimisation function directly)
        tol: array eg. np.array([x1,x2]) tolerance of minimisation in each dimension
        step: float, gradient calculation step length    
    """
    errt = abserror2(minimum,tol,step,True)
    erra = abserror2(minimum,tol,step,False)
    return errt, erra


#------------------------------------------------------------------------------
"""Integrating the fit function"""
"""
dx = 0.01
x = np.arange(-50,50,dx)
tautrial = np.arange(0.1,1.0,0.1)
sigmatrial = np.arange(0.1,1.0,0.1)
for i in tautrial:
    for j in sigmatrial:
        y = fitfunction(x,i,j)
        integral = y*dx
        print "Integral for tau = " +str(i) + " and sigma = " +str(j)+ " : " + str(np.sum(integral))
"""
#------------------------------------------------------------------------------
"""Plotting histogram with fitting"""
"""
plt.figure()
plt.hist(t, bins, alpha=0.2, normed=True)
plt.title("Uncertainty Distribution")
plt.xlabel("Uncertainty/ ps")
plt.ylabel("Frequency", fontsize = 18)

plt.plot(np.array(sorted(t)),fitfunction(np.array(sorted(t)), 0.2, 0.3), label = 'tau = 0.2, sigma = 0.3')
plt.plot(np.array(sorted(t)),fitfunction(np.array(sorted(t)), 0.4, 0.3), label = 'tau = 0.4, sigma = 0.3')
plt.plot(np.array(sorted(t)),fitfunction(np.array(sorted(t)), 0.6, 0.3),'-m', label = 'tau = 0.6, sigma = 0.3')
plt.xlabel("Decay Time/ ps", fontsize = 18)
plt.ylabel("Fit function")
plt.legend()
plt.show()
"""

#------------------------------------------------------------------------------
"""Plotting 1D minimisation path"""
"""
plt.figure()
for i in range(0, tau.size):
    NLLsums[i] = NLL(t,tau[i],resol)
plt.plot(tau, NLLsums)

print "Minimum located at: " + str(minimiser(x, tol,10000))
print "Absolute Error: " + str(abserror(minimiser(x,tol,10000), h,10000))
print "Standard Deviation:" + str(stdev(10000))

plt.plot(np.asarray(xplot),yplot,'-co', markevery = [0,-1], linewidth=0.5) 
xplot = []
yplot = []

plt.plot(tau, NLLsums, '-m')
plt.title("Negative Log Likelihood Minimisation")
plt.xlabel("Decay Time/ ps", fontsize = 18)
plt.ylabel("NLL", fontsize = 18)
plt.xlim((0,1.0))
plt.ylim((6000,14000))
plt.show()
"""
#------------------------------------------------------------------------------
"""Fit accuracy"""
"""
plt.figure()
datapoints= np.arange(1,10002,200)
accuracyabs = []
accuracystd = []
test = []
test1 = []

for data in datapoints:
    x0 = 0.1
    x2 = 0.6
    x = np.array([x0,(x0+x2)/2,x2])
    o = minimiser(x,tol,data)
    print o
    accuracystd.append(stdev(data))
    accuracyabs.append(abserror(o,step,data))
for i in range(len(accuracystd)):
    test.append(1/accuracystd[i]**2)
    test1.append(1/accuracyabs[i]**2)
plt.plot(accuracystd,datapoints,'-r', label='Curvature Stdv')
plt.plot(accuracyabs,datapoints, '-g', label='Absolute Stdv')
plt.title('Sample uncertainty')
plt.xlabel('Standard Deviation /ps', fontsize = 18)
plt.ylabel('Sample size', fontsize = 18)
plt.legend()
plt.ylim((0,10000))

plt.figure()
plt.plot(test,datapoints, 'or', label = 'Standard Deviation')
plt.plot(test1,datapoints, 'og', label = 'Absolute Uncertainty')
plt.plot(np.unique(test1), np.poly1d(np.polyfit(test1, datapoints, 1))(np.unique(test1)),'-g')
plt.plot(np.unique(test), np.poly1d(np.polyfit(test, datapoints, 1))(np.unique(test)),'-r')
plt.title('Sample uncertainty', fontsize = 18)
plt.xlabel('(1/Standard Deviation$^2$) /ps$^{-2}$', fontsize = 18)
plt.ylabel('Sample size', fontsize = 18)
plt.ylim((0,10000))
plt.legend(loc=4)
plt.show()

m,b = np.polyfit(test1, datapoints, 1)
equation1 = 'Abs Uncertainty: y = ' + str(round(m,4)) + 'x' ' + ' + str(round(b,4))
a,c = np.polyfit(test, datapoints, 1)
equation2 = 'Standard dev: y = ' + str(round(a,4)) + 'x' ' + ' + str(round(c,4))
print equation1
print equation2
"""
#------------------------------------------------------------------------------
"""Plotting 2D minimistation paths"""
"""
plt.figure()   
counter = 0
start = []
while counter < trials:  
    start.append([rand.uniform(0.1,1.0),rand.uniform(0.05,1.0)])
    counter = counter + 1
for i in start:
    print("\n")
    print "For starting point " + str(i)
    if gradient == True:
        print "Gradient method" + str(minimiser2d(i,h,alpha, toler))
        if error == True:
            print " +/- " + str(abserror2D(minimiser2d(i, h, alpha, toler)[0],tol,step))
    if newton == True:
        print "Newton method" + str(minimiser2dnewton(i,h, toler))
        if error == True:
            print " +/- " + str(abserror2D(minimiser2dnewton(i, h, toler)[0],tol,step))
    if quasi == True:
        print "Quasi-Newton method" + str(minimiser2dquasi(i,h,alpha, toler))
        if error == True:
            print " +/- " + str(abserror2D(minimiser2dquasi(i, h, alpha, toler)[0],tol,step))
    plt.plot(tauplot,aplot,'-ro', markevery = [0,-1], linewidth = 0.1)
    plt.plot(tauplot1,aplot1,'-bo',markevery = [0,-1], linewidth = 0.1)
    plt.plot(tauplot2,aplot2,'-go', markevery = [0,-1], linewidth = 0.1)
    tauplot = []
    aplot = []
    tauplot1 = []
    aplot1 = []
    tauplot2 = []
    aplot2 = []

contourplot()
"""
#------------------------------------------------------------------------------
"""Plotting the allowed Newton Method starting point region"""
"""
for i in np.arange(0,1.01,0.05):
    for j in np.arange(0,1.01,0.05):
        with np.errstate(divide='ignore'):
            with np.errstate(invalid='ignore'):
                with np.errstate(over='ignore'):
                    x = minimiser2dnewton([i,j],h,toler)
                    if x[0][1] < 1.0 and x[0][1] > 0.0 and x[0][0] > 0.0:
                        print x, (i,j)
                        plt.plot(i,j,'gs', alpha = 0.4)
                        tauplot1 = []
                        aplot1 = []
                      
contourplot()
"""
