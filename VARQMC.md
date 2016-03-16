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
alpha =  0.05 , <E> =  -2.87229008493 var(E) =  0.176021424779
alpha =  0.075 , <E> =  -2.87507574409 var(E) =  0.153108213894
alpha =  0.1 , <E> =  -2.87644176413 var(E) =  0.136218721972
alpha =  0.125 , <E> =  -2.87866648628 var(E) =  0.122412620692
alpha =  0.15 , <E> =  -2.87855217954 var(E) =  0.112162720412
alpha =  0.175 , <E> =  -2.87733166555 var(E) =  0.103333112745
alpha =  0.2 , <E> =  -2.87655266183 var(E) =  0.096896981427
alpha =  0.25 , <E> =  -2.87516014275 var(E) =  0.0886909592825
```

```python
>>> ## Alpha minimizer
```

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
>>> alpha = np.zeros(shape=(numbalpha+1,))
>>> alpha[0] = 1.2
>>> beta = 0.6
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
...     alpha[i+1] = alpha[i] -((i+1)**(-beta))*dEdalpha
...
>>> alpha = np.delete(alpha, numbalpha, axis = 0)
alpha =  1.2 , <E> =  0.703624264887 var(E) =  0.500329546151
alpha =  0.779555003235 , <E> =  0.549827112383 var(E) =  0.109042582789
alpha =  0.564001590469 , <E> =  0.506882995164 var(E) =  0.00651357765691
alpha =  0.508777836661 , <E> =  0.500377713395 var(E) =  0.000138791498002
alpha =  0.500940836778 , <E> =  0.499974303502 var(E) =  1.89959632486e-06
alpha =  0.500038738976 , <E> =  0.499999214768 var(E) =  3.19644250754e-09
alpha =  0.50000505477 , <E> =  0.49999984551 var(E) =  5.46017188935e-11
alpha =  0.500000972011 , <E> =  0.500000018634 var(E) =  1.87473753215e-12
alpha =  0.500000290105 , <E> =  0.499999987063 var(E) =  1.82635107352e-13
alpha =  0.500000080256 , <E> =  0.499999997894 var(E) =  1.39803706504e-14
alpha =  0.50000002517 , <E> =  0.500000000189 var(E) =  1.2266926831e-15
alpha =  0.500000010476 , <E> =  0.499999999992 var(E) =  2.2812025165e-16
alpha =  0.50000000419 , <E> =  0.500000000092 var(E) =  3.41773110856e-17
alpha =  0.500000001927 , <E> =  0.500000000057 var(E) =  6.66684739698e-18
alpha =  0.500000001003 , <E> =  0.499999999971 var(E) =  2.06735560544e-18
alpha =  0.500000000471 , <E> =  0.499999999996 var(E) =  4.34499192653e-19
alpha =  0.50000000024 , <E> =  0.500000000002 var(E) =  1.03139751483e-19
alpha =  0.500000000136 , <E> =  0.500000000004 var(E) =  3.50934759808e-20
alpha =  0.500000000075 , <E> =  0.500000000001 var(E) =  1.07814672942e-20
alpha =  0.500000000042 , <E> =  0.5 var(E) =  3.56893423268e-21
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
>>> alpha = np.zeros(shape=(numbalpha+1,))
>>> alpha[0] = 0.7
>>> beta = 0.6
>>> N = 400
>>> steps = 30000
>>> d = 0.05 #movement size
>>> meanEn = np.zeros(shape=(numbalpha,))
>>> varE = np.zeros(shape=(numbalpha,))
...
...
>>> for i in range(numbalpha):
...     x = np.random.uniform(-1,1,(N,3))
...     Energy = np.zeros(shape=(steps,N))
...     lnpsi = np.zeros(shape=(steps,N))
...     Energy, lnpsi = simulate_hydrogen_atom_min(alpha[i],steps,x)
...     meanEn[i] = np.mean(Energy[4000:,:])
...     varE[i] = np.var(Energy[4000:,:])
...
...     print("alpha = ",alpha[i],", <E> = ", meanEn[i], "var(E) = ", varE[i])
...
...     meanlnpsi = np.mean(lnpsi[4000:,:])
...     meanEtimeslnpsi = np.mean(lnpsi[4000:,:]*Energy[4000:,:])
...     dEdalpha = 2*(meanEtimeslnpsi-meanEn[i]*meanlnpsi)
...     alpha[i+1] = alpha[i] -((i+1)**(-beta))*dEdalpha
...
>>> alpha = np.delete(alpha, numbalpha, axis = 0)
alpha =  0.7 , <E> =  -0.488425121979 var(E) =  0.0535864985323
alpha =  0.907692862344 , <E> =  -0.501974354319 var(E) =  0.00722230726736
alpha =  0.937279977592 , <E> =  -0.503444755849 var(E) =  0.00393425394877
alpha =  0.95150979846 , <E> =  -0.501440346503 var(E) =  0.00226437364314
alpha =  0.955150079483 , <E> =  -0.500388759757 var(E) =  0.00180209885257
alpha =  0.958050565313 , <E> =  -0.502054306976 var(E) =  0.00207269005911
alpha =  0.961872307088 , <E> =  -0.501160489915 var(E) =  0.00141691605791
alpha =  0.963298308992 , <E> =  -0.502442018081 var(E) =  0.00135902745328
alpha =  0.966278627158 , <E> =  -0.501980036858 var(E) =  0.00120548961465
alpha =  0.968452263326 , <E> =  -0.501432412668 var(E) =  0.00104162133578
alpha =  0.969940146234 , <E> =  -0.501786836668 var(E) =  0.000938258858578
alpha =  0.97283738231 , <E> =  -0.502152063244 var(E) =  0.000810039384146
alpha =  0.973654115647 , <E> =  -0.501114864581 var(E) =  0.000925426312754
alpha =  0.974910238481 , <E> =  -0.501465784055 var(E) =  0.000645964158584
alpha =  0.97672147083 , <E> =  -0.501401998704 var(E) =  0.000576265823781
alpha =  0.977769147755 , <E> =  -0.500721920624 var(E) =  0.000502526926938
alpha =  0.978943229862 , <E> =  -0.500996472554 var(E) =  0.000450960713468
alpha =  0.980246617808 , <E> =  -0.5014472737 var(E) =  0.000417948367572
alpha =  0.981070469812 , <E> =  -0.500345811938 var(E) =  0.000347110724603
alpha =  0.981500949124 , <E> =  -0.500817142623 var(E) =  0.00036159897885
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
>>> alpha = np.zeros(shape=(numbalpha+1,))
>>> alpha[0] = 1.2
>>> beta = 0.6
>>> N = 400
>>> steps = 10000
>>> d = 0.3 #movement size
>>> meanEn = np.zeros(shape=(numbalpha,))
>>> varE = np.zeros(shape=(numbalpha,))
...
...
>>> for i in range(numbalpha):
...     X = np.random.uniform(-2,2,(2*N,3))
...     Energy = np.zeros(shape=(steps,N))
...     lnpsi = np.zeros(shape=(steps,N))
...     Energy, lnpsi = simulate_helium_vector_min(alpha[i],steps,X,d)
...     meanEn[i] = np.mean(Energy[4000:,:])
...     varE[i] = np.var(Energy[4000:,:])
...
...     print("alpha = ",alpha[i],", <E> = ", meanEn[i], "var(E) = ", varE[i])
...
...     meanlnpsi = np.mean(lnpsi[4000:,:])
...     meanEtimeslnpsi = np.mean(lnpsi[4000:,:]*Energy[4000:,:])
...     dEdalpha = 2*(meanEtimeslnpsi-meanEn[i]*meanlnpsi)
...     alpha[i+1] = alpha[i] -((i+1)**(-beta))*dEdalpha
...
>>> alpha = np.delete(alpha, numbalpha, axis = 0)
alpha =  1.2 , <E> =  -2.81080791673 var(E) =  0.228563474769
alpha =  1.15971635351 , <E> =  -2.81342554033 var(E) =  0.220487069726
alpha =  1.13222615974 , <E> =  -2.82020993145 var(E) =  0.214194045027
alpha =  1.11008139892 , <E> =  -2.82603445107 var(E) =  0.209166445006
alpha =  1.09103872417 , <E> =  -2.82094035002 var(E) =  0.204710267929
alpha =  1.07421042789 , <E> =  -2.82291419958 var(E) =  0.202296422115
alpha =  1.05885588317 , <E> =  -2.8206525684 var(E) =  0.199683214721
alpha =  1.04463862716 , <E> =  -2.82275368919 var(E) =  0.197084726997
alpha =  1.03134580139 , <E> =  -2.82986753047 var(E) =  0.19378844249
alpha =  1.01873567098 , <E> =  -2.82752902034 var(E) =  0.190960994076
alpha =  1.00683364509 , <E> =  -2.82496295033 var(E) =  0.189717279609
alpha =  0.995438578827 , <E> =  -2.82474222376 var(E) =  0.187850542409
alpha =  0.984459509987 , <E> =  -2.82926414715 var(E) =  0.185203559006
alpha =  0.973893283581 , <E> =  -2.81930317836 var(E) =  0.184169674038
alpha =  0.963721207093 , <E> =  -2.82866572328 var(E) =  0.181016642671
alpha =  0.953828744026 , <E> =  -2.82493769034 var(E) =  0.179174943821
alpha =  0.944292383731 , <E> =  -2.82609216377 var(E) =  0.176502079365
alpha =  0.935054101109 , <E> =  -2.82957928399 var(E) =  0.176140550548
alpha =  0.925930761949 , <E> =  -2.83272309818 var(E) =  0.173159012946
alpha =  0.917070724981 , <E> =  -2.82858984252 var(E) =  0.169321993791
```

```python

```
