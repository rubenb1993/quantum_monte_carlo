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
...             E_old = alpha + x_old*x_old *(0.5-2*(alpha*alpha))
...             x_new = x_old + (random() - 0.5)*d
...             E_new = alpha + x_new*x_new *(0.5-2*(alpha*alpha))
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
>>> alpha = [0.4,0.45,0.55,0.6]
>>> N = 400
>>> steps = 30000
>>> d = 0.05 #movement size
...
>>> # Initiate random x vector
... x = np.random.uniform(-1,1,(N,1))
>>> size = x.shape
>>> Energy = np.zeros(shape=(steps,N))
>>> meanEn = np.zeros(shape=(len(alpha),))
>>> varE = np.zeros(shape=(len(alpha),))
...
>>> for i in range(len(alpha)):
...     x = np.random.uniform(-1,1,(N,1))
...     Energy = np.zeros(shape=(steps,N))
...     Energy = simulate_harmonic(alpha[i],steps,x)
...     meanEn[i] = np.mean(Energy[4000:,:])
...     varE[i] = np.var(Energy[4000:,:])
...     print("alpha = ",alpha[i],", <E> = ", meanEn[i], "var(E) = ", varE[i])
```

```python

```

```python

```
