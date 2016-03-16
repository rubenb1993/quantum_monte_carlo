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
...         r12_old = np.linalg.norm(X[0:N,:] - X[N:,:],axis=1)
...         r12_new = np.linalg.norm(X_new[0:N,:] - X_new[N:,:],axis=1)
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
>>> d = 0.3
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
alpha =  0.05 , <E> =  -2.87061628463 var(E) =  0.175217486358
alpha =  0.075 , <E> =  -2.87618637107 var(E) =  0.153836920106
alpha =  0.1 , <E> =  -2.87821278424 var(E) =  0.13619809781
alpha =  0.125 , <E> =  -2.87905988678 var(E) =  0.122925304434
alpha =  0.15 , <E> =  -2.87820442748 var(E) =  0.111534430571
alpha =  0.175 , <E> =  -2.87605583605 var(E) =  0.103171651965
alpha =  0.2 , <E> =  -2.87737875063 var(E) =  0.096748373754
alpha =  0.25 , <E> =  -2.8727049448 var(E) =  0.0882921842801
```

## Alpha minimizer

```python
>>> def simulate_harmonic_min(alpha,steps,x):
...     for j in range(steps):
...         x_new = x + (np.random.rand(len(x)) - 0.5)*d
...         p = (np.exp(-alpha*x_new*x_new) / np.exp(-alpha*x*x))**2
...         m = p > np.random.rand(len(x))
...         x = x_new*m + x*~m
...         Energy[j,:] = alpha + x*x *(0.5 - 2*(alpha*alpha))
...         lnpsi[j,:] = -x*x
...     return Energy, lnpsi
```

```python
>>> #%%timeit
... numbalpha = 10
>>> alpha = np.zeros(shape=(numbalpha+1,))
>>> alpha[0] = 1.2
>>> gamma = 1
>>> N = 400
>>> steps = 30000
>>> d = 0.05 #movement size
>>> meanEn = np.zeros(shape=(numbalpha,))
>>> varE = np.zeros(shape=(numbalpha,))
...
>>> for i in range(numbalpha):
...     x = np.random.uniform(-1,1,(N))
...     Energy = np.zeros(shape=(steps,N))
...     lnpsi = np.zeros(shape=(steps,N))
...     Energy, lnpsi = simulate_harmonic_min(alpha[i],steps,x)
...     meanEn[i] = np.mean(Energy[4000:,:])
...     varE[i] = np.var(Energy[4000:,:])
...     print("alpha = ",alpha[i],", <E> = ", meanEn[i], "var(E) = ", varE[i])
...
...     meanlnpsi = np.mean(lnpsi[4000:,:])
...     meanEtimeslnpsi = np.mean(lnpsi[4000:,:]*Energy[4000:,:])
...     dEdalpha = 2*(meanEtimeslnpsi-meanEn[i]*meanlnpsi)
...     alpha[i+1] = alpha[i] -gamma*dEdalpha
...
>>> alpha = np.delete(alpha, numbalpha, axis = 0)
alpha =  1.2 , <E> =  0.694710000411 var(E) =  0.498871935951
alpha =  0.780779885755 , <E> =  0.548714887257 var(E) =  0.104460897534
alpha =  0.490301875193 , <E> =  0.499820237501 var(E) =  0.000180616921021
alpha =  0.5091081624 , <E> =  0.499980195235 var(E) =  0.000171248187129
alpha =  0.490476247239 , <E> =  0.499680997751 var(E) =  0.000160632890517
alpha =  0.507504978699 , <E> =  0.499936982457 var(E) =  0.000115545993494
alpha =  0.49222375173 , <E> =  0.50009285066 var(E) =  0.000119856721108
alpha =  0.507757728666 , <E> =  0.500087447469 var(E) =  0.000112582440749
alpha =  0.493357150244 , <E> =  0.500095677153 var(E) =  8.9207589968e-05
alpha =  0.506876068888 , <E> =  0.500290102105 var(E) =  8.66020181856e-05
```

```python

```

```python

```
