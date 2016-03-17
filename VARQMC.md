```python
>>> import matplotlib.pyplot as plt
>>> import numpy as np
>>> from scipy.optimize import fsolve
...
>>> %matplotlib inline
```

```python
>>> def simulate_harmonic(alpha,steps,x):
...     "A variational monte carlo method for a harmonic oscilator"
...     for j in range(steps):
...         x_new = x + (np.random.rand(len(x)) - 0.5)*d
...         p = (np.exp(-alpha*x_new*x_new) / np.exp(-alpha*x*x))**2
...         m = p > np.random.rand(len(x)) #Vector with acceptences
...         x = x_new*m + x*~m #new positions
...         Energy[j,:] = alpha + x*x *(0.5 - 2*(alpha*alpha))
...     return Energy
```

```python
>>> def simulate_helium_vector(alpha,steps,X):
...     """A variational Monte Carlo simulation for a helium atom.
...     Based on theory of ``Computational Physics'' by J.M. Thijssen, chapter 12.2 (2nd edition)
...     X is an (2N,3) matrix with particle pairs (x,y,z) position. Particles are paired with i and N+i
...     alpha is the trial variable for the trial wave function of the form exp(-alpha * r)
...     steps is the amount of steps taken by the walkers
...     Energy (steps, N) is the energy of each particle pair at timestep j
...     """
...     for j in range(steps):
...         X_new = X + (np.random.rand(2*N,3) - 0.5) * d
...         r_old = np.linalg.norm(X,axis=1)
...         r_new = np.linalg.norm(X_new,axis=1)
...         r12_old = np.linalg.norm(X[0:N,:] - X[N:,:],axis=1)
...         r12_new = np.linalg.norm(X_new[0:N,:] - X_new[N:,:],axis=1)
...
...         psi_fact_old = 1 + alpha*r12_old
...         psi_fact_new = 1 + alpha*r12_new
...
...         psi_old = np.exp(-2*r_old[0:N] - 2*r_old[N:]) * np.exp(r12_old/(2*psi_fact_old))
...         psi_new= np.exp(-2*r_new[0:N] - 2*r_new[N:]) * np.exp(r12_new/(2*psi_fact_new))
...
...         p = (psi_new/psi_old) ** 2
...         m = p > np.random.rand(N) #Vector with acceptance of new position {size= (200,1)}
...         m = np.transpose(np.tile(m,(3,2))) #make from m a 400,3 matrix by repeating m
...         X = X_new*(m) + X*~(m)
...
...         #Make normalization vector of (200,3) to normalize each particle pair
...         r1_length = np.transpose(np.tile(np.linalg.norm(X[0:N,:], axis=1), (3,1)))
...         r2_length = np.transpose(np.tile(np.linalg.norm(X[N:,:], axis=1), (3,1)))
...
...         #Vectors for energy calculation
...         r1r2_diff = X[0:N,:] - X[N:,:]
...         r1r2_diff_hat = X[0:N,:]/r1_length - X[N:,:]/r2_length
...
...         #recurring factors in energy calculation
...         r12 = np.linalg.norm(X[0:N,:] - X[N:,:], axis=1)
...         psi_fact = 1 + alpha*r12
...
...         #dot product (r1_hat - r2_hat) * (r1 - r2)
...         dot_product = np.sum(r1r2_diff_hat*r1r2_diff, axis=1)
...
...         Energy[j,:] = -4 + dot_product / (r12*psi_fact**2) - 1 / (r12*psi_fact**3) - 1 / (4*psi_fact**4) + 1 / r12
...     return Energy
```

```python
>>> def simulate_hydrogen_atom(alpha,steps,x):
...     "a variational monte carlo method for the hydrogen atom"
...     for j in range(steps):
...         x_new = x + (np.random.random_sample(np.shape(x)) - 0.5)*d
...         p = (np.exp(-alpha*np.linalg.norm(x_new, axis=1)) / np.exp(-alpha*np.linalg.norm(x, axis=1))) ** 2
...         m = (p > np.random.rand(N)).reshape(-1,1) #acceptance vector
...         x = x_new*m + x*~m
...         Energy[j,:] = -1/np.linalg.norm(x, axis=1) - alpha/2*(alpha - 2/np.linalg.norm(x, axis=1))
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
...     Energy = simulate_helium_vector(alpha[i],steps,X)
...     meanEn[i] = np.mean(Energy[4000:,:])
...     varE[i] = np.var(Energy[4000:,:])
...
...     print("alpha = ",alpha[i],", <E> = ", meanEn[i], "var(E) = ", varE[i])
alpha =  0.05 , <E> =  -2.87115747721 var(E) =  0.175024515771
alpha =  0.075 , <E> =  -2.87671725997 var(E) =  0.153744624121
alpha =  0.1 , <E> =  -2.87637077663 var(E) =  0.136753147217
alpha =  0.125 , <E> =  -2.87629164014 var(E) =  0.122201267971
alpha =  0.15 , <E> =  -2.8785236435 var(E) =  0.111547873615
alpha =  0.175 , <E> =  -2.87847111638 var(E) =  0.103380955243
alpha =  0.2 , <E> =  -2.87653363486 var(E) =  0.0969477551553
alpha =  0.25 , <E> =  -2.87391030154 var(E) =  0.0887472087972
```

## Hydrogen Molecule

```python
>>> def simulate_hydrogen_molecule(s,a,beta,steps,X):
...     """Variational Quantum mechanics procedure for calculating the expected energy from parameters s and beta
...     Using the Coulomb cusp condition, and the method described in chapter 12.2 from Computational Physics by J.M. Thijssen
...     steps: integer amount of steps the walkers walk
...     X: (2*N,3) matrix of (x,y,z) position of all walker pairs (pairs: i and N+i) with N amount of walkers
...     """
...     a = fsolve(f,0.1)
...     r_1 = np.ones([1,3])
...     r_2 = np.zeros([1,3])
...     r_12 = r_1 - r_2
...     r_12_abs = np.linalg.norm(r_12,axis=1)
...     r_1L = r_1 + s/2*np.array([1,0,0])
...     r_1L_abs = np.linalg.norm(r_1L, axis=1)
...     r_1R = r_1 - s/2*np.array([1,0,0])
...     r_1R_abs = np.linalg.norm(r_1R, axis=1)
...     r_2L = r_2 + s/2*np.array([1,0,0])
...     r_2L_abs = np.linalg.norm(r_2L, axis=1)
...     r_2R = r_2 - s/2*np.array([1,0,0])
...     r_2R_abs = np.linalg.norm(r_2R, axis=1)
...     phi_1L, phi_1R, phi_2L, phi_2R = np.exp(-np.array([r_1L_abs, r_1R_abs, r_2L_abs, r_2R_abs])/a)
...     phi_1 = phi_1L + phi_1R
...     phi_2 = phi_2L + phi_2R
...     r_1L_hat = r_1L/r_1L_abs
...     r_1R_hat = r_1R/r_1R_abs
...     r_2L_hat = r_2L/r_2L_abs
...     r_2R_hat = r_2R/r_2R_abs
...     r_12_hat = r_12/r_12_abs
...
...     psi_jastrow = np.exp(r_12_abs/(2*(1+beta*r_12_abs)))
...     psi =
...     return Energy
```

```python
>>> beta = 1
>>> N = 400
>>> steps = 30000
>>> d = 0.3
>>> s = 1
>>> def f(a):
...     """Coulumb cusp condition analytical expression
...     """
...     return 1/(1 + np.exp(-s/a)) - a
...
>>> a = fsolve(f,0.1)
>>> print(a)
>>> print(f(a))
[ 0.78218829]
[ -1.11022302e-16]
```

```python
>>> X = np.random.uniform(-2,2,(2*N,3))
>>> Energy = simulate_hydrogen_atom(s,a,beta,steps,X)
>>> meanEn = np.mean(Energy[4000:,:])
>>> varE = np.var(Energy[4000:,:])
...
>>> print("beta = ",beta,", <E> = ", meanEn, "var(E) = ", varE)
alpha =  0.05 , <E> =  -2.87115747721 var(E) =  0.175024515771
alpha =  0.075 , <E> =  -2.87671725997 var(E) =  0.153744624121
alpha =  0.1 , <E> =  -2.87637077663 var(E) =  0.136753147217
alpha =  0.125 , <E> =  -2.87629164014 var(E) =  0.122201267971
alpha =  0.15 , <E> =  -2.8785236435 var(E) =  0.111547873615
alpha =  0.175 , <E> =  -2.87847111638 var(E) =  0.103380955243
alpha =  0.2 , <E> =  -2.87653363486 var(E) =  0.0969477551553
alpha =  0.25 , <E> =  -2.87391030154 var(E) =  0.0887472087972
```

```python
>>> from types import SimpleNamespace
```

```python
>>> a = np.zeros((2,3))
>>> a
array([[ 0.,  0.,  0.],
       [ 0.,  0.,  0.]])
```

```python
>>> b = np.ones(2)
```

```python

```

```python
>>> a
array([[ 1.,  0.,  0.],
       [ 1.,  0.,  0.]])
```

```python

```

```python
>>> b
array([ 1.,  1.])
```

```python
>>> a[:, 0] += b
>>> a
```

```python
>>> #a = fsolve(f,0.1)
... a = 1
>>> pos_walker = np.random.rand(N,3,2,2)
>>> #pos_walker = np.ones([N,3,2,2])
... pos_walker[:,:,:,1] = pos_walker[...,0] + (np.random.rand(N,3,2) - 0.5)*d
>>> offset_array = np.append(s/2*np.ones([N,1,2,2]),np.zeros([N,2,2,2]), axis=1)
>>> left_right_array = np.append(pos_walker + offset_array, pos_walker - offset_array, axis=2)
>>> phi_1L, phi_2L, phi_1R, phi_2R = np.transpose(np.exp(np.linalg.norm(left_right_array,axis=1)/-a), axes=[1, 0, 2])
>>> phi_1 = phi_1L + phi_1R
>>> phi_2 = phi_2L + phi_2R
>>> r_12 = np.diff(pos_walker, axis=2)
>>> r_12_abs = np.linalg.norm(r_12, axis=1)
>>> psi_jastrow = np.squeeze(np.exp(r_12_abs/(2*(1+beta*r_12_abs))))
>>> psi = phi_1*phi_2*psi_jastrow
>>> p = (psi[:,1]/psi[:,0]) ** 2
>>> mask = p > np.random.rand(N)
>>> mask_walker = np.tile(mask,(2,3,1)).T
>>> mask_left_right = np.tile(mask_walker,(1,1,2))
>>> mask_r_abs = np.tile(mask,(1,1)).T
>>> mask_r_12 = np.tile(mask,(1,3,1)).T
>>> def apply_mask(mat, mask):
...     return mat[..., 0] * mask + mat[..., 1] * ~mask
...
>>> phi_1L = apply_mask(phi_1L, mask).T
>>> phi_2L = apply_mask(phi_2L, mask).T
>>> phi_1R = apply_mask(phi_1R, mask).T
>>> phi_2R = apply_mask(phi_2R, mask).T
>>> phi_1 = phi_1L + phi_1R
>>> phi_2 = phi_2L + phi_2R
>>> pos_walker[...,0] = apply_mask(pos_walker, mask_walker)
>>> r_12_abs = apply_mask(r_12_abs, mask_r_abs).T
>>> r_1L, r_2L, r_1R, r_2R = apply_mask(left_right_array, mask_left_right).T
>>> r_12 = np.transpose(apply_mask(r_12, mask_r_12), axes = [1,0,2])
>>> r_12_hat = np.transpose(r_12/r_12_abs, axes = [1,0,2])
>>> def normalize(vec):
...     absvec = np.linalg.norm(vec)
...     return absvec, vec/absvec
...
...
>>> r_1L_abs, r_1L_hat = normalize(r_1L)
>>> r_2L_abs, r_2L_hat = normalize(r_2L)
>>> r_1R_abs, r_1R_hat = normalize(r_1R)
>>> r_2R_abs, r_2R_hat = normalize(r_2R)
...
>>> dot_1 = (phi_1L*r_1L_hat + phi_1R*r_1R_hat)/phi_1 - (phi_2L*r_2L_hat + phi_2R*r_2R_hat)/phi_2
>>> dot_2 = r_12/(2*a*(1 + beta*r_12_abs)**2)
>>> print(dot_1.shape,dot_2.shape)
...
>>> # dot_product = np.sum(1*r1r2_diff, axis=1)
...
...
... Energy = (-1/a**2 + (phi_1L/r_1L_abs + phi_1R/r_1R_abs)/(a*phi_1) + (phi_2L/r_2L_abs + phi_2R/r_2R_abs)/(a*phi_2) - \
...          1/r_1L_abs - 1/r_1R_abs - 1/r_2L_abs - 1/r_2R_abs + 1/r_12_abs - ((4*beta + 1)*r_12_abs + 4)/(4*r_12_abs*(1 + beta*r_12_abs)**4) + \
...          np.dot((phi_1L*r_1L_hat + phi_1R*r_1R_hat)/phi_1 - (phi_2L*r_2L_hat + phi_2R*r_2R_hat)/phi_2, r_12/(2*a*(1 + beta*r_12_abs)**2)))
...
>>> print(Energy.shape)
(3, 400) (3, 400, 400)
(3, 3, 400)
```

```python

```

```python
>>> phi_1L[:, 1] == phi_1L[:, 1]*m
```

```python

```
