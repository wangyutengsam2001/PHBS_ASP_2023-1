#!/usr/local/bin/python
# -*- coding: gbk -*-
#============================================================
# TEST1.PY                     -- by Dr. ZhuoQing 2020-07-10
#
# Note:
#============================================================

# from headm import *

#################################################################
#   Function InvLap(t,omega,sigma,nint), numerically inverts a  #
#   Laplace transform F(s) into f(t) using the Fast Fourier     #
#   Transform (FFT) algorithm for a specific time "t", an       #
#   upper frequency limit "omega", a real parameter "sigma"     #
#   and the number of integration intervals "nint" .            #
#                                                               #
#   Function F(s) is defined in separate as Fs(s) (see code     #
#   below). Fs(s) has to be changed accordingly everytime the   #
#   user wants to invert a different function.                  #
#                                                               #
#   I suggest to use omega>100 and nint=50*omega. The higher    #
#   the values for omega, the more accurate the results will be #
#   in general, but at the expense of longer processing times.  #
#                                                               #
#   Sigma is a real number which must be a little bigger than   #
#   the real part of rightmost pole of the function F(s). For   #
#   example, F(s) = 1/s + 1/(s-2) + 1/(s+1) has poles for s=0,  #
#   s=2 and s=-1. Hence, sigma must be made equal to, say,      #
#   2.05 so as to keep all poles at the left of this value.     #
#   The analytical inverse for this simple function is          #
#   f(t) = 1 + exp(-t) + exp(2t). For t=1.25, omega=200,        #
#   nint=10000 and sigma=2.05, the numerical inversion yields   #
#   f(1.25) ~= 13.456844516, or -0.09% away from the actual     #
#   analytical result, 13.468998757 (results truncated to 9     #
#   decimal places). This is the example used in this code.     #
#                                                               #
#   Creator: Fausto Arinos de Almeida Barbuto (Calgary, Canada) #
#   Date: May 18, 2002                                          #
#   E-mail: fausto_barbuto@yahoo.ca                             #
#                                                               #
#   Reference:                                                  #
#   Huddleston, T. and Byrne, P: "Numerical Inversion of        #
#   Laplace Transforms", University of South Alabama, April     #
#   1999 (found at http://www.eng.usouthal.edu/huddleston/      #
#   SoftwareSupport/Download/Inversion99.doc)                   #
#                                                               #
#   Usage: invoke InvLap(t,omega,sigma,nint), for t>0.          #
#                                                               #
#################################################################
#   We need cmath because F(s) is a function operating on the
#   complex argument s = a + bj
from math import ceil
from cmath import *

#   *** Driver InvLap function  ***
def InvLap(t,omega,sigma,nint,Fs):
#   Sanity check on some parameters.
    omega = ceil(omega)
    nint = ceil(nint)

    if omega <= 0:
       omega = 200

    if nint <= 0:
        nint = 10000

    return (trapezoid(t,omega,sigma,nint,Fs))

#   *** Function trapezoid computes the numerical inversion. ***
def trapezoid(t,omega,sigma,nint,Fs):
    sum = 0.0
    delta = float(omega)/nint
    wi = 0.0

#   The for-loop below computes the FFT Inversion Algorithm.
#   It is in fact the trapezoidal rule for numerical integration.
    for i in range(1,(nint+1)):
        witi = complex(0,wi*t)

        wf = wi + delta
        wfti = complex(0,wf*t)

        fi = (exp(witi)*Fs(complex(sigma,wi))).real
        ff = (exp(wfti)*Fs(complex(sigma,wf))).real
        sum = sum + 0.5*(wf-wi)*(fi+ff)
        wi = wf

    return ((sum*exp(sigma*t)/pi).real)

#   *** The Laplace function F(s) is defined here.  ***
# def Fs(λ, k=k, θ=theta, ε=epsilon, T=texp, v0=v0):
#     λ /= T
#     γ = np.sqrt(k**2 + 2*λ*ε**2)
#     b_0T = 2*λ*(np.exp(γ*T)-1)/((γ+k)*(np.exp(γ*T)-1) + 2*γ)
#     α = γ + k
#     β = γ - k

#     a_0T = (k*θ/ε**2) * (k-γ)*T

#     dt = 1e-3
#     n_dt = int(T / dt)
#     t = np.arange(0, T, dt)
#     for k in range(n_dt-1):
#         part1 = α*np.exp(γ*(T-t[k+1]))
#         part2 = β*np.exp(-γ*dt)
#         a_0T -= (2*k*θ/ε**2) * np.log((part1+part2) / (part1+β))

#     L = np.exp(a_0T - b_0T*v0)
#     return L

#   Function InvLap(t,omega,sigma,nint) is invoked.
# print(InvLap(1.25, 200, 3.05, 10000))

#------------------------------------------------------------
#        END OF FILE : TEST1.PY
#============================================================
