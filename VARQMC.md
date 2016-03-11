```python
>>> import matplotlib.pyplot as plt
>>> import numpy as np
>>> from random import random
>>> from math import exp
>>> from numba import jit
...
>>> %matplotlib inline
```

```python
>>> @jit
... def simulate_harmonic(alpha,steps,x):
...     for j in range(steps):
...         for i in range(len(x)):
...             x_old = x[i]
...             x_new = x_old + (random() - 0.5)*d
...             p = (exp(-alpha*x_new*x_new) / exp(-alpha*x_old*x_old))**2
...             if p >= 1:
...                 x[i] = x_new
...             elif p > random():
...                 x[i] = x_new
...             else:
...                 x[i] = x_old
...             Energy[j,i] = alpha + x[i]*x[i] *(0.5-2*(alpha*alpha))
...     return Energy
```

```python
>>> def simulate_helium_vector(alpha,steps,X):
...     shape = X.shape
...     for j in range(steps):
...         X_new = X + (np.random.rand(2*N,3) - 0.5) * d
...         r_old = np.linalg.norm(X,axis=1)
...         r_new = np.linalg.norm(X_new,axis=1)
...         r12_old = np.linalg.norm(X[0:N,:] - X[N:,:])
...         r12_new = np.linalg.norm(X_new[0:N,:] - X_new[N:,:])
...
...         psi_fact_old = 1 + alpha * r12_old
...         psi_fact_new = 1 + alpha * r12_new
...
...         psi_old = np.exp(-2*r_old[0:N] - 2*r_old[N:]) * np.exp(r12_old/(2*psi_fact_old))
...         psi_new= np.exp(-2*r_new[0:N] - 2*r_new[N:]) * np.exp(r12_new/(2*psi_fact_new))
...
...         p = (psi_new/psi_old)**2
...         m = p>np.random.rand(N)
...         m = np.transpose(np.tile(m,(3,2))) #make from m a 400,3 matrix by repeating m
...         X = X_new*(m) + X*~(m)
...
...         r1r2_diff = X[0:N,:] - X[N:,:]
...         r1_length = np.transpose(np.tile(np.linalg.norm(X[0:N,:],axis=1),(3,1))) #200,3 length vector to normalize r1
...         r2_length = np.transpose(np.tile(np.linalg.norm(X[N:,:],axis=1),(3,1))) #200,3 length vecotr to normalize r2
...         r1r2_diff_hat = X[0:N,:]/r1_length - X[N:,:]/r2_length
...
...         r12 = np.linalg.norm(X[0:N,:] - X[N:,:],axis=1)
...         psi_fact = 1 + alpha*r12
...
...         dot_product = np.sum(r1r2_diff_hat * r1r2_diff,axis=1)
...
...         Energy[j,:] = -4 + dot_product/(r12*psi_fact**2) - 1/(r12*psi_fact**3) - 1/(4*psi_fact**4) + 1/r12
...     return Energy
```

```python
>>> def simulate_helium(alpha,steps,X):
...     for j in range(steps):
...         for i in range(N):
...             x_old = X[i,:]
...             x_new = x_old + (np.random.uniform(0,1,(1,6)) - 0.5) * d
...
...             r1_old = np.linalg.norm(x_old[0:3])
...             r2_old = np.linalg.norm(x_old[3:])
...
...             r1_new = np.linalg.norm(x_new[0:3])
...             r2_new = np.linalg.norm(x_new[3:])
...
...             r12_old = x_old[0:3] - x_old[3:]
...             r12_old_abs = np.linalg.norm(r12_old)
...             r12_old_hat = r12_old/r12_old_abs
...
...             r12_new = x_new[:,0:3] - x_new[:,3:]
...             r12_new_abs = np.linalg.norm(r12_new)
...             r12_new_hat = r12_new[0,:]/r12_new_abs
...
...             psi_fact_old = 1 + alpha*r12_old_abs
...             psi_fact_new = 1 + alpha*r12_new_abs
...
...             psi_old = exp(-2*r1_old)*exp(-2*r2_old) * exp(r12_old_abs/(2*psi_fact_old))
...             psi_new = exp(-2*r1_new)*exp(-2*r2_new) * exp(r12_new_abs/(2*psi_fact_new))
...             p = (psi_new/psi_old)**2
...             if p > random():
...                 X[i,:] = x_new
...                 dot_new = np.dot(r12_new_hat,r12_new[0,:])
...                 Energy[j,i] = -4 + dot_new/(r12_new_abs * psi_fact_new ** 2) - \
...                                     1/(r12_new_abs * (psi_fact_new**3)) - 1/(4*(psi_fact_new**4)) + 1/r12_new_abs
...             else:
...                 X[i,:] = x_old
...                 Energy[j,i] = -4 + np.dot(r12_old_hat,r12_old)/(r12_old_abs * psi_fact_old ** 2) - \
...                                     1/(r12_old_abs * psi_fact_old**3) - 1/(4*psi_fact_old**4) + 1/r12_old_abs
...     return Energy
```

## 1D harmonic oscilator

```python
>>> alpha = [0.4,0.45,0.55,0.6]
>>> N = 4
>>> steps = 30000
>>> d = 0.5 #movement size
...
>>> # Initiate random x vector
... x = np.random.uniform(-1,1,(N))
>>> Energy = np.zeros(shape=(steps,N))
>>> meanEn = np.zeros(shape=(len(alpha),))
>>> varE = np.zeros(shape=(len(alpha),))
>>> n = np.zeros(shape = (len(alpha),))
...
>>> for i in range(len(alpha)):
...     x = np.random.uniform(-1,1,(N))
...     Energy = np.zeros(shape=(steps,N))
...     Energy = simulate_harmonic(alpha[i],steps,x)
...     meanEn[i] = np.mean(Energy[4000:,:])
...     varE[i] = np.var(Energy[4000:,:])
...     print("alpha = ",alpha[i],", <E> = ", meanEn[i], "var(E) = ", varE[i])
alpha =  0.4 , <E> =  0.513156140604 var(E) =  0.0248828562207
alpha =  0.45 , <E> =  0.502200683745 var(E) =  0.00535360079856
alpha =  0.55 , <E> =  0.504218832321 var(E) =  0.00420340150802
alpha =  0.6 , <E> =  0.509519918784 var(E) =  0.0165193730982
```

## Helium Atom

```python
>>> alpha = [0.05,0.075,0.10,0.125,0.15,0.175,0.20,0.25]
>>> N = 400
>>> steps = 30000
>>> d = 1.5
...
>>> meanEn = np.zeros(shape=(len(alpha),))
>>> varE = np.zeros(shape=(len(alpha),))
...
...
...
>>> for i in range(len(alpha)):
...     X = np.random.uniform(-10,10,(2*N,3))
...     Energy = np.zeros(shape=(steps,N))
...     Energy = simulate_helium_vector(alpha[i],steps,X)
...     meanEn[i] = np.mean(Energy[4000:,:])
...     varE[i] = np.var(Energy[4000:,:])
...
...     print("alpha = ",alpha[i],", <E> = ", meanEn[i], "var(E) = ", varE[i])
alpha =  0.05 , <E> =  -2.99548597678 var(E) =  0.189113512296
alpha =  0.075 , <E> =  -2.97676399691 var(E) =  0.168381189406
alpha =  0.1 , <E> =  -2.95823907086 var(E) =  0.151126750389
alpha =  0.125 , <E> =  -2.94095237902 var(E) =  0.136653830221
alpha =  0.15 , <E> =  -2.92592585629 var(E) =  0.125050566022
alpha =  0.175 , <E> =  -2.91114252923 var(E) =  0.115004291746
alpha =  0.2 , <E> =  -2.90007251029 var(E) =  0.106951557626
alpha =  0.25 , <E> =  -2.87860371232 var(E) =  0.095512205894
```

```python

(3, 400)
```

```python

[[ 0.  1.  2.]
 [ 3.  4.  5.]
 [ 6.  7.  8.]]
[   5.   50.  149.]
```
