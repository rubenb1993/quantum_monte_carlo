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
...             E_old = alpha + x_old*x_old*(0.5 - 2*(alpha*alpha))
...             x_new = x_old + (random() - 0.5)*d
...             E_new = alpha + x_new*x_new *(0.5 - 2*(alpha*alpha))
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
...         Energy[j,:] = alpha + x*x *(0.5 - 2*(alpha*alpha))
...     return Energy
```

```python
>>> alpha = 0.4
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
alpha =  0.4 , <E> =  0.510900453409 var(E) =  0.0263659449371
alpha =  0.4 , <E> =  0.513247966289 var(E) =  0.0251702541311
alpha =  0.4 , <E> =  0.50451251969 var(E) =  0.0206442379429
alpha =  0.4 , <E> =  0.509817295917 var(E) =  0.0242520019181
1 loop, best of 3: 2.6 s per loop
```

```python
>>> %%timeit
... Energy = simulate_harmonic_numba(alpha,steps,x)
... meanEn = np.mean(Energy[4000:,:])
... varE = np.var(Energy[4000:,:])
... print("alpha = ",alpha,", <E> = ", meanEn, "var(E) = ", varE)
```

```python

```
