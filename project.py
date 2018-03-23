# -*- coding: utf-8 -*-
"""
Created on Wed Nov 23 10:32:18 2016

@author: Priyanka Mocherla
@version: 2.7

This code contains functions to analyse a particle data set to extract the lifetime of a particle and the amount of background in the data set. Derived log likelihood functions and used numerical methods to minimise the function and extract the error in the data.

Also includes the code required to complete the project tasks.

Functions in this module:
    - fitfunction
    - NLL
    - function
    - minimiser
    - abserror
    - stdev
    - fitfunction_bg
    - NLL_bg
    - function_bg
    - minimiser2d
    - minimiser2dnewton
    - minimiser2dquasi


"""
#------------------------ Functions and Data Prep -----------------–––#
import numpy as np
import matplotlib.pyplot as plt
import math as m
import random as rand

#--------------------------Preparing the data-------------------------#
data = np.loadtxt("lifetime.txt")

x0 = 0.2
x2 = 0.6
tol = 0.0001
bins = np.linspace(-2, 6, 160)
tau = np.arange(0.1,1.01,0.05)
a = np.arange(0.05,1.01,0.05)
tautest = 0.2
atest = 0.2
sigma = 0.01

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
y = np.zeros(x.size)

Z = np.zeros((a.size,tau.size))

initial = [0.1,0.9]
toler = [tol,tol]
h = 0.0001
alpha = 0.00001
aplot = []
tauplot = []
aplot1 = []
tauplot1 = []

#------------------------------ Functions --------------------------------#

"""NOT including background radiation"""

def fitfunction(time, lifetime, resolution):
    for i in range(0, t.size):
        fitdata[i] = (1/(2*lifetime))*np.exp((resolution**2/(2*lifetime**2))-(time[i]/lifetime))*m.erfc((1/np.sqrt(2))*((resolution/lifetime)-(time[i]/resolution)))
    return fitdata
    
def NLL(time, lifetime, resolution):
    for i in range(0, t.size):
        NLLdata[i] = - np.log((1/(2*lifetime))*np.exp((resolution[i]**2/(2*lifetime**2))-(time[i]/lifetime))*m.erfc((1/np.sqrt(2))*((resolution[i]/lifetime)-(time[i]/resolution[i]))))
    return np.sum(NLLdata)
    
def function(time, tau, resolution):
    return NLL(time, tau, resolution)
    
def minimiser(x, tol):
    for i in range(x.size):
        y[i] = function(t, x[i],resol)
    x3 = 0.5*((x[2]**2-x[1]**2)*y[0]+(x[0]**2-x[2]**2)*y[1]+(x[1]**2-x[0]**2)*y[2])/((x[2]-x[1])*y[0]+(x[0]-x[2])*y[1]+(x[1]-x[0])*y[2])
    xplot.extend(x)
    yplot.extend(y)
    while (abs(np.amax(y) - np.amin(y)) > tol) and (function(t, x3, resol) < np.amax(y)):
        x[y.argmax()] = x3
        y[y.argmax()] = function(t, x3, resol)
        xplot.append(x3)
        yplot.append(function(t,x3, resol))
        x3 = 0.5*((x[2]**2-x[1]**2)*y[0]+(x[0]**2-x[2]**2)*y[1]+(x[1]**2-x[0]**2)*y[2])/((x[2]-x[1])*y[0]+(x[0]-x[2])*y[1]+(x[1]-x[0])*y[2])
        continue
    else:
        xplot.append(x[y.argmin()])
        yplot.append(y[y.argmin()])
        return x[y.argmin()],y[y.argmin()]
        
def abserror(minimum, step):
    NLLplus = minimum[1] + 0.5
    startplus = minimum[0]
    while NLLplus > NLL(t, startplus, resol):
        startplus = startplus + step
        continue
    else:
        return (startplus - minimum[0]) 
        
def stdev():
    minimiser(x, tol)      
    d2P2 = (2/((x[1]-x[0])*(x[2]-x[0])*(x[2]-x[1])))*(y[0]*(x[2]-x[1])+y[1]*(x[0]-x[2])+y[2]*(x[1]-x[0]))
    stdev = 1/np.sqrt(d2P2)
    return stdev


"""With background radiation"""
def fitfunction_bg(time, lifetime, fraction, resolution):
    for i in range(0,t.size):
        fitdata[i] = (fraction*(1/(2*lifetime))*np.exp((resolution[i]**2/(2*lifetime**2))-(time[i]/lifetime))*m.erfc((1/np.sqrt(2))*((resolution[i]/lifetime)-(time[i]/resolution)))) + np.exp((-time[i]**2)/(2*resolution[i]**2))*(1-fraction)/(resolution[i]*np.sqrt(2*np.pi))
    return fitdatabg

def NLL_bg(time, lifetime, fraction, resolution):
    for i in range(0,t.size):
        NLLdatabg[i] = - np.log((fraction*(1/(2*lifetime))*np.exp((resolution[i]**2/(2*lifetime**2))-(time[i]/lifetime))*m.erfc((1/np.sqrt(2))*((resolution[i]/lifetime)-(time[i]/resolution[i])))) + np.exp((-time[i]**2)/(2*resolution[i]**2))*(1-fraction)/(resolution[i]*np.sqrt(2*np.pi)))
    return np.sum(NLLdatabg)
    
def function_bg(x,y):
    return NLL_bg(t, x, y, resol)

def minimiser2d(start, step, alpha, tol):
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
    tauplot1.append(start[0])
    aplot1.append(start[1])
    H = np.identity((2))
    grad = np.zeros(2)
    test = np.zeros(2)
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

#---------------------------------- Project Code ---------------------------------#

#plt.figure()
#plt.hist(data[:,0], bins, alpha=0.8, normed=True)
#plt.title("Decay Time Distribution")
#plt.xlabel("Decay Time/ ps")
#plt.ylabel("Frequency")

#plt.figure()
#plt.plot(t,fitfunction(t, tautest, sigma))
#plt.xlabel("Decay Time/ ps")
#plt.ylabel("Fit function")
#plt.show()

for i in range(0, tau.size):
    NLLsums[i] = NLL(t,tau[i],resol)
plt.figure()
plt.plot(tau, NLLsums)
plt.xlabel("Lifetime Estimate/ ps")
plt.ylabel("NLL")

counter = 0
start = []

while counter < 1:
    start.append(np.array((rand.uniform(0.1,1.0),rand.uniform(0.1,1.0),rand.uniform(0.1,1.0))))
    counter = counter + 1
print start
for i in start:
    print minimiser(i, tol)
    print abserror(minimiser(x,tol), 1e-5)
    print stdev()
    plt.plot(np.asarray(xplot),yplot,'-go', markevery = [0,-1], linewidth=0.2)
    xplot = []
    yplot = []
plt.xlabel("Decay Time/ ps")
plt.ylabel("NLL")
plt.show()

"""With background"""
#plt.figure()
#plt.hist(data[:,0], bins, alpha=0.8, normed=True)
#plt.title("Decay Time Distribution")
#plt.xlabel("Decay Time/ ps")
#plt.ylabel("Frequency")

#plt.figure()
#plt.scatter(t,fitfunction_bg(t, tautest, atest, sigma))
#plt.xlabel("Decay Time/ ps")
#plt.ylabel("Fit function")
#plt.show()


plt.figure()
counter = 0
start = []

while counter < 5:  
    start.append([rand.uniform(0.1,1.0),rand.uniform(0.05,1.0)])
    counter = counter + 1
print start  
for i in start:
    print minimiser2d(i,h,alpha, toler)
    print minimiser2dquasi(i,h,alpha, toler)
    plt.plot(tauplot,aplot,'-ro', markevery = [0,-1], linewidth = 0.1)
    plt.plot(tauplot1,aplot1,'-bo', markevery = [0,-1], linewidth = 0.1)
    tauplot = []
    aplot = []
    tauplot1 = []
    aplot1 = []
    
X, Y = np.meshgrid(tau, a)
for i in range(a.size):
    print i
    for j in range(tau.size):
        Z[i][j] = NLL_bg(t,tau[j],a[i],resol)
plt.contour(X, Y, Z, levels = np.arange(6000,22000,1000))
plt.colorbar()
plt.title('Minimising Fraction of signal and Lifetime of sample')
plt.xlabel('Lifetime/ ps')
plt.ylabel('Fraction of signal')
plt.ylim((0.05,1.0))
plt.xlim((0.1,1.0))
plt.show()

