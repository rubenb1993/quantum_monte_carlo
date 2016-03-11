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
... def simulate_harmonic_numba(alpha,steps,x):
...     for j in range(steps):
...         for i in range(len(x)):
...             x_old = x[i]
...             x_new = x_old + (random() - 0.5)*d
...             p = (exp(-alpha*x_new*x_new) / exp(-alpha*x_old*x_old))**2
...             if p > random():
...                 x[i] = x_new
...             else:
...                 x[i] = x_old
...             Energy[j,i] = alpha + x[i]*x[i] *(0.5 - 2*(alpha*alpha))
...     return Energy
```

```python
>>> def simulate_harmonic(alpha,steps,x):
...     for j in range(steps):
...         x_new = x + (np.random.rand(len(x)) - 0.5)*d
...         p = (np.exp(-alpha*x_new*x_new) / np.exp(-alpha*x*x))**2
...         m = p > np.random.rand(len(x))
...         x = x_new*m + x*~m
...         #Energy[j,:] = alpha + x*x *(0.5 - 2*(alpha*alpha))#harmonic oscillator
...         Energy[j,:] = -1/x -alpha/2*(alpha - 2/x)#harmonic oscillator
...     return Energy
```

```python
>>> alpha = 0.8
>>> N = 400 # 400
>>> steps = 30000 # 30000
>>> d = 0.05 #movement size
...
>>> # Initiate random x vector
... x = np.random.uniform(-1,1,(N))
>>> Energy = np.zeros(shape=(steps,N))
```

```python
>>> %%timeit
... Energy = simulate_harmonic(alpha,steps,x)
... meanEn = np.mean(Energy[4000:,:])
... varE = np.var(Energy[4000:,:])
... print("alpha = ",alpha,", <E> = ", meanEn, "var(E) = ", varE)
alpha =  0.8 , <E> =  5.41990592754 var(E) =  253329534.611
alpha =  0.8 , <E> =  0.724882826951 var(E) =  11940888.9738
alpha =  0.8 , <E> =  0.686928729276 var(E) =  3877241.15404
alpha =  0.8 , <E> =  -10.9155286413 var(E) =  1086063933.53
1 loop, best of 3: 2.73 s per loop
```

```python
>>> %%timeit
... Energy = simulate_harmonic_numba(alpha,steps,x)
... meanEn = np.mean(Energy[4000:,:])
... varE = np.var(Energy[4000:,:])
... print("alpha = ",alpha,", <E> = ", meanEn, "var(E) = ", varE)
alpha =  0.4 , <E> =  0.514238784676 var(E) =  0.0258623796311
alpha =  0.4 , <E> =  0.512987288177 var(E) =  0.0246134071106
alpha =  0.4 , <E> =  0.509713052555 var(E) =  0.0237407845676
alpha =  0.4 , <E> =  0.516960323286 var(E) =  0.0279417000937
1 loop, best of 3: 1min 1s per loop
```

```python

```
