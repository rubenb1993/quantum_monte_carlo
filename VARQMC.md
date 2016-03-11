```python
>>> import matplotlib.pyplot as plt
>>> import numpy as np
...
>>> %matplotlib inline
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
>>> def simulate_hydrogen_atom(alpha,steps,x):
...     for j in range(steps):
...         x_new = x + (np.random.random_sample(np.shape(x)) - 0.5)*d
...         p = (np.exp(-alpha*np.linalg.norm(x_new, axis=1)) / np.exp(-alpha*np.linalg.norm(x, axis=1)))**2
...         m = (p > np.random.rand(N)).reshape(-1,1)
...         x = x_new*m + x*~m
...         #Energy[j,:] = alpha + x*x *(0.5 - 2*(alpha*alpha))#harmonic oscillator
...         Energy[j,:] = -1/np.linalg.norm(x, axis=1) -alpha/2*(alpha - 2/np.linalg.norm(x, axis=1))#hydrogen atom
...     return Energy
```

```python
>>> alpha = 1.1
>>> N = 400 # 400
>>> steps = 30000 # 30000
>>> d = 0.05 #movement size
...
>>> # Initiate random x vector
... x = np.random.uniform(-1,1,(N,3))
>>> Energy = np.zeros(shape=(steps,N))
```

```python
>>> #%%timeit
... Energy = simulate_hydrogen_atom(alpha,steps,x)
>>> meanEn = np.mean(Energy[4000:,:])
>>> varE = np.var(Energy[4000:,:])
>>> print("alpha = ",alpha,", <E> = ", meanEn, "var(E) = ", varE)
alpha =  1.1 , <E> =  -0.488854446753 var(E) =  0.0134825221291
```

```python
>>> alpha = [0.4,0.45,0.5,0.55,0.6]
>>> N = 400
>>> steps = 30000
>>> d = 0.05 #movement size
>>> # Initiate random x vector
... meanEn = np.zeros(shape=(len(alpha),))
>>> varE = np.zeros(shape=(len(alpha),))
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
