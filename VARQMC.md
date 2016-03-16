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
alpha =  0.4 , <E> =  0.51285452681 var(E) =  0.0267334062996
alpha =  0.45 , <E> =  0.503660199428 var(E) =  0.00577682870384
alpha =  0.5 , <E> =  0.5 var(E) =  0.0
alpha =  0.55 , <E> =  0.502063011635 var(E) =  0.00465870951288
alpha =  0.6 , <E> =  0.5065228474 var(E) =  0.0175632139657
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
alpha =  0.05 , <E> =  -2.87229008493 var(E) =  0.176021424779
alpha =  0.075 , <E> =  -2.87507574409 var(E) =  0.153108213894
alpha =  0.1 , <E> =  -2.87644176413 var(E) =  0.136218721972
alpha =  0.125 , <E> =  -2.87866648628 var(E) =  0.122412620692
alpha =  0.15 , <E> =  -2.87855217954 var(E) =  0.112162720412
alpha =  0.175 , <E> =  -2.87733166555 var(E) =  0.103333112745
alpha =  0.2 , <E> =  -2.87655266183 var(E) =  0.096896981427
alpha =  0.25 , <E> =  -2.87516014275 var(E) =  0.0886909592825
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
... numbalpha = 20
>>> alpha = 1.2
>>> beta = 0.6
>>> N = 400
>>> steps = 30000
>>> d = 0.05 #movement size
>>> varE = 10
>>> i = 0
>>> while varE > 0.01 and i < numbalpha:
...     x = np.random.uniform(-1,1,(N))
...     Energy = np.zeros(shape=(steps,N))
...     lnpsi = np.zeros(shape=(steps,N))
...     Energy, lnpsi = simulate_harmonic_min(alpha,steps,x)
...     meanEn = np.mean(Energy[4000:,:])
...     varE = np.var(Energy[4000:,:])
...     print("alpha = ",alpha,", <E> = ", meanEn, "var(E) = ", varE)
...
...     meanlnpsi = np.mean(lnpsi[4000:,:])
...     meanEtimeslnpsi = np.mean(lnpsi[4000:,:]*Energy[4000:,:])
...     dEdalpha = 2*(meanEtimeslnpsi-meanEn*meanlnpsi)
...     alpha -= alpha - ((i+1)**(-beta))*dEdalpha
...     i += 1
...
>>> print("End result: alpha = ",alpha,", <E> = ", meanEn, 'var(E) = ', varE)
alpha =  1.2 , <E> =  0.713789892327 var(E) =  0.469265522051
alpha =  0.805659225167 , <E> =  0.556528769304 var(E) =  0.124714953222
alpha =  0.599485564841 , <E> =  0.506300250735 var(E) =  0.0171733492035
alpha =  0.495902823234 , <E> =  0.500187585504 var(E) =  3.63418961187e-05
End result: alpha =  0.501778906218 , <E> =  0.500187585504 var(E) =  3.63418961187e-05
```

```python
>>> def simulate_hydrogen_atom_min(alpha,steps,x):
...     for j in range(steps):
...         x_new = x + (np.random.random_sample(np.shape(x)) - 0.5)*d
...         p = (np.exp(-alpha*np.linalg.norm(x_new, axis=1)) / np.exp(-alpha*np.linalg.norm(x, axis=1)))**2
...         m = (p > np.random.rand(N)).reshape(-1,1)
...         x = x_new*m + x*~m
...         Energy[j,:] = -1/np.linalg.norm(x, axis=1) -alpha/2*(alpha - 2/np.linalg.norm(x, axis=1))
...         lnpsi[j,:] = -np.linalg.norm(x)
...     return Energy, lnpsi
```

```python
>>> #%%timeit
... numbalpha = 20
>>> alpha = 0.7
>>> beta = 0.6
>>> N = 400
>>> steps = 30000
>>> d = 0.05 #movement size
>>> varE = 10
>>> i = 0
...
>>> while varE > 0.001 and i < numbalpha:
...     x = np.random.uniform(-1,1,(N,3))
...     Energy = np.zeros(shape=(steps,N))
...     lnpsi = np.zeros(shape=(steps,N))
...     Energy, lnpsi = simulate_hydrogen_atom_min(alpha,steps,x)
...     meanEn = np.mean(Energy[4000:,:])
...     varE = np.var(Energy[4000:,:])
...
...     print("alpha = ",alpha,", <E> = ", meanEn, "var(E) = ", varE)
...
...     meanlnpsi = np.mean(lnpsi[4000:,:])
...     meanEtimeslnpsi = np.mean(lnpsi[4000:,:]*Energy[4000:,:])
...     dEdalpha = 2*(meanEtimeslnpsi-meanEn*meanlnpsi)
...     alpha -= ((i+1)**(-beta))*dEdalpha
...     i += 1
...
>>> print("End result: alpha = ",alpha,", <E> = ", meanEn, 'var(E) = ', varE)
alpha =  0.7 , <E> =  -0.484607664436 var(E) =  0.0520042652239
alpha =  0.950513592372 , <E> =  -0.502979011176 var(E) =  0.00259511090811
alpha =  0.960569933486 , <E> =  -0.501006210651 var(E) =  0.00147155585563
alpha =  0.96826958044 , <E> =  -0.502104445138 var(E) =  0.00106939365245
alpha =  0.970115200392 , <E> =  -0.500945643954 var(E) =  0.00100783752149
alpha =  0.974705272081 , <E> =  -0.501639390298 var(E) =  0.00068897093722
End result: alpha =  0.975648814032 , <E> =  -0.501639390298 var(E) =  0.00068897093722
```

```python
>>> def simulate_helium_vector_min(alpha,steps,X,d):
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
...         lnpsi[j,:] = -2*(r12*r12)/(4*(1+alpha*r12)*(1+alpha*r12))
...     return Energy, lnpsi
```

```python
>>> numbalpha = 20
>>> alpha = 0.3
>>> beta = 0.6
>>> N = 400
>>> steps = 10000
>>> d = 0.3 #movement size
...
>>> varE = 10
>>> i = 0
...
>>> while varE > 0.01 and i < numbalpha:
...     X = np.random.uniform(-2,2,(2*N,3))
...     Energy = np.zeros(shape=(steps,N))
...     lnpsi = np.zeros(shape=(steps,N))
...     Energy, lnpsi = simulate_helium_vector_min(alpha,steps,X,d)
...     meanEn = np.mean(Energy[4000:,:])
...     varE = np.var(Energy[4000:,:])
...
...     print("alpha = ",alpha,", <E> = ", meanEn, "var(E) = ", varE)
...
...     meanlnpsi = np.mean(lnpsi[4000:,:])
...     meanEtimeslnpsi = np.mean(lnpsi[4000:,:]*Energy[4000:,:])
...     dEdalpha = 2*(meanEtimeslnpsi-meanEn*meanlnpsi)
...     alpha -=  ((i+1)**(-beta))*dEdalpha
...     i += 1
...
>>> print("End result: alpha = ",alpha,", <E> = ", meanEn, 'var(E) = ', varE)
alpha =  0.3 , <E> =  -2.87215493533 var(E) =  0.0851903521687
alpha =  0.231872401931 , <E> =  -2.87377270434 var(E) =  0.0906146196936
alpha =  0.197044404797 , <E> =  -2.87583655031 var(E) =  0.0975278326562
End result: alpha =  0.178374389803 , <E> =  -2.87583655031 var(E) =  0.0975278326562
```

```python

```
