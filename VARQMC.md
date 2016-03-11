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
>>> def simulate_helium_vector(alpha,steps,X,d):
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
>>> def simulate_hydrogen_atom(alpha,steps,x):
...     for j in range(steps):
...         x_new = x + (np.random.random_sample(np.shape(x)) - 0.5)*d
...         p = (np.exp(-alpha*np.linalg.norm(x_new, axis=1)) / np.exp(-alpha*np.linalg.norm(x, axis=1)))**2
...         m = (p > np.random.rand(N)).reshape(-1,1)
...         x = x_new*m + x*~m
...         Energy[j,:] = -1/np.linalg.norm(x, axis=1) -alpha/2*(alpha - 2/np.linalg.norm(x, axis=1))
...     return Energy
```

## 1D harmonic oscilator

```python
>>> #%%timeit
... alpha = [0.4,0.45,0.5,0.55,0.6]
>>> N = 400
>>> steps = 30000
>>> d = 0.05 #movement size
>>> meanEn = np.zeros(shape=(len(alpha),))
>>> varE = np.zeros(shape=(len(alpha),))
...
>>> for i in range(len(alpha)):
...     x = np.random.uniform(-1,1,(N))
...     Energy = np.zeros(shape=(steps,N))
...     Energy = simulate_harmonic(alpha[i],steps,x)
...     meanEn[i] = np.mean(Energy[4000:,:])
...     varE[i] = np.var(Energy[4000:,:])
...     print("alpha = ",alpha[i],", <E> = ", meanEn[i], "var(E) = ", varE[i])
alpha =  0.4 , <E> =  0.506460538552 var(E) =  0.022707115056
alpha =  0.45 , <E> =  0.500555176808 var(E) =  0.0050657545702
alpha =  0.5 , <E> =  0.5 var(E) =  0.0
alpha =  0.55 , <E> =  0.502103176214 var(E) =  0.00465720593767
alpha =  0.6 , <E> =  0.510190280355 var(E) =  0.0164588352981
```

## Hydrogen Atom

```python
>>> #%%timeit
... alpha = [0.8,0.9,1,1.1,1.2]
>>> N = 400
>>> steps = 30000
>>> d = 0.05 #movement size
>>> meanEn = np.zeros(shape=(len(alpha),))
>>> varE = np.zeros(shape=(len(alpha),))
...
>>> for i in range(len(alpha)):
...     x = np.random.uniform(-1,1,(N,3))
...     Energy = np.zeros(shape=(steps,N))
...     Energy = simulate_hydrogen_atom(alpha[i],steps,x)
...     meanEn[i] = np.mean(Energy[4000:,:])
...     varE[i] = np.var(Energy[4000:,:])
...     print("alpha = ",alpha[i],", <E> = ", meanEn[i], "var(E) = ", varE[i])
alpha =  0.8 , <E> =  -0.500951463889 var(E) =  0.0301581243297
alpha =  0.9 , <E> =  -0.503837775564 var(E) =  0.00946968379623
alpha =  1 , <E> =  -0.5 var(E) =  0.0
alpha =  1.1 , <E> =  -0.491888181923 var(E) =  0.0127833303647
alpha =  1.2 , <E> =  -0.467792154199 var(E) =  0.0602454548027
```

## Helium Atom

```python
>>> alpha = [0.05,0.075,0.10,0.125,0.15,0.175,0.20,0.25]
>>> N = 400
>>> steps = 30000
>>> d = 0.1
...
>>> meanEn = np.zeros(shape=(len(alpha),))
>>> varE = np.zeros(shape=(len(alpha),))
...
...
...
>>> for i in range(len(alpha)):
...     X = np.random.uniform(-2,2,(2*N,3))
...     Energy = np.zeros(shape=(steps,N))
...     Energy = simulate_helium_vector(alpha[i],steps,X,d)
...     meanEn[i] = np.mean(Energy[4000:,:])
...     varE[i] = np.var(Energy[4000:,:])
...
...     print("alpha = ",alpha[i],", <E> = ", meanEn[i], "var(E) = ", varE[i])
alpha =  0.05 , <E> =  -2.95232385864 var(E) =  0.208017640869
alpha =  0.075 , <E> =  -2.94151484022 var(E) =  0.183915577893
alpha =  0.1 , <E> =  -2.9260539173 var(E) =  0.164984569561
alpha =  0.125 , <E> =  -2.90774480284 var(E) =  0.147805417769
alpha =  0.15 , <E> =  -2.89417130552 var(E) =  0.133078844921
alpha =  0.175 , <E> =  -2.88272199571 var(E) =  0.120518915052
alpha =  0.2 , <E> =  -2.87544795127 var(E) =  0.112528599967
alpha =  0.25 , <E> =  -2.85985545676 var(E) =  0.0986507347475
```

```python
>>> alpha = [0.05, 0.075, 0.1, 0.125, 0.15, 0.175, 0.2, 0.25]
>>> N = 400
>>> steps = 30000
>>> d = [0.0015, 0.0015, 0.02, 0.0225, 0.1, 0.25, 0.25, 1]
>>> Xrange = [0.5, 0.5, 0.5, 0.5, 1, 2, 2, 2]
>>> meanEn = np.zeros(shape=(len(alpha),))
>>> varE = np.zeros(shape=(len(alpha),))
...
...
...
>>> for i in range(len(alpha)):
...     X = np.random.uniform(-Xrange[i],Xrange[i],(2*N,3))
...     Energy = np.zeros(shape=(steps,N))
...     Energy = simulate_helium_vector(alpha[i],steps,X,d[i])
...     meanEn[i] = np.mean(Energy[4000:,:])
...     varE[i] = np.var(Energy[4000:,:])
...
...     print("alpha = ",alpha[i],", <E> = ", meanEn[i], "var(E) = ", varE[i])
alpha =  0.05 , <E> =  -2.90431953315 var(E) =  0.217867226938
alpha =  0.075 , <E> =  -2.8705912082 var(E) =  0.187755874208
alpha =  0.1 , <E> =  -2.90583105399 var(E) =  0.17355659988
alpha =  0.125 , <E> =  -2.87084527599 var(E) =  0.151615907644
alpha =  0.15 , <E> =  -2.89473207706 var(E) =  0.132211286959
alpha =  0.175 , <E> =  -2.88673662107 var(E) =  0.12001119664
alpha =  0.2 , <E> =  -2.8766466008 var(E) =  0.110835168778
alpha =  0.25 , <E> =  -2.87172814924 var(E) =  0.0967602312541
```

```python

```
