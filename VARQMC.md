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
... x = np.random.uniform(-1,1,(N,1))
>>> Energy = np.zeros(shape=(steps,N))
>>> meanEn = np.zeros(shape=(len(alpha),))
>>> varE = np.zeros(shape=(len(alpha),))
>>> n = np.zeros(shape = (len(alpha),))
...
>>> for i in range(len(alpha)):
...     x = np.random.uniform(-1,1,(N,1))
...     Energy = np.zeros(shape=(steps,N))
...     Energy,n[i] = simulate_helium(alpha[i],steps,x)
...     meanEn[i] = np.mean(Energy[4000:,:])
...     varE[i] = np.var(Energy[4000:,:])
...     print("alpha = ",alpha[i],", <E> = ", meanEn[i], "var(E) = ", varE[i])
```

## Helium Atom

```python
>>> alpha = [0.25]
>>> N = 200
>>> steps = 30000
>>> d = 0.1
...
>>> meanEn = np.zeros(shape=(len(alpha),))
>>> varE = np.zeros(shape=(len(alpha),))
...
...
...
>>> for i in range(len(alpha)):
...     X = np.random.uniform(-1,1,(N,6))
...     Energy = np.zeros(shape=(steps,N))
...     Energy = simulate_helium(alpha[i],steps,X)
...     meanEn[i] = np.mean(Energy[4000:,:])
...     print(meanEn[i])
```

```python
>>> fig = plt.figure()
>>> ax = fig.add_subplot(111)
>>> ax.plot(Energy[4000:,10:20])
```

```python

```
