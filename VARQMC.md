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
<<<<<<< HEAD
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
>>> diffmeanEn = 10
>>> meanEn = 0
>>> i = 0
...
>>> while diffmeanEn > 0.0001 and i < numbalpha:
...     x = np.random.uniform(-1,1,(N))
...     Energy = np.zeros(shape=(steps,N))
...     lnpsi = np.zeros(shape=(steps,N))
...     Energy, lnpsi = simulate_harmonic_min(alpha,steps,x)
...
...     meanEnNew = np.mean(Energy[4000:,:])
...     varE = np.var(Energy[4000:,:])
...     diffmeanEn = np.absolute(meanEnNew - meanEn)
...     meanEn = meanEnNew
...     print("alpha = ",alpha,", <E> = ", meanEn, "var(E) = ", varE)
...
...     meanlnpsi = np.mean(lnpsi[4000:,:])
...     meanEtimeslnpsi = np.mean(lnpsi[4000:,:]*Energy[4000:,:])
...     dEdalpha = 2*(meanEtimeslnpsi-meanEn*meanlnpsi)
...     alpha -= ((i+1)**(-beta))*dEdalpha
...     i += 1
...
>>> print("End result: alpha = ",alpha,", <E> = ", meanEn, 'var(E) = ', varE)
alpha =  1.2 , <E> =  0.697068254543 var(E) =  0.493908254026
alpha =  0.784951047037 , <E> =  0.558389266278 var(E) =  0.101114692857
alpha =  0.602754798757 , <E> =  0.508341027584 var(E) =  0.016607971568
alpha =  0.526938466568 , <E> =  0.501556409501 var(E) =  0.00127767964761
alpha =  0.506835104718 , <E> =  0.500107161141 var(E) =  9.23756053681e-05
alpha =  0.501724506103 , <E> =  0.499995271393 var(E) =  5.93865014345e-06
alpha =  0.50055127395 , <E> =  0.500004761534 var(E) =  6.12967077504e-07
End result: alpha =  0.500205516533 , <E> =  0.500004761534 var(E) =  6.12967077504e-07
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
>>> diffmeanEn = 10
>>> meanEn = 0
...
>>> while diffmeanEn > 0.001 and i < numbalpha:
...     x = np.random.uniform(-1,1,(N,3))
...     Energy = np.zeros(shape=(steps,N))
...     lnpsi = np.zeros(shape=(steps,N))
...     Energy, lnpsi = simulate_hydrogen_atom_min(alpha,steps,x)
...     meanEnNew = np.mean(Energy[4000:,:])
...     varE = np.var(Energy[4000:,:])
...     diffmeanEn = np.absolute(meanEnNew-meanEn)
...     meanEn = meanEnNew
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
alpha =  0.7 , <E> =  -0.482601188276 var(E) =  0.0463859311278
alpha =  0.901913485647 , <E> =  -0.500261459423 var(E) =  0.00879691149083
alpha =  0.927155159321 , <E> =  -0.504120862701 var(E) =  0.00513690493756
alpha =  0.93782386961 , <E> =  -0.503056481043 var(E) =  0.00378557467176
alpha =  0.946192948957 , <E> =  -0.501215194201 var(E) =  0.00282414308947
alpha =  0.951280818907 , <E> =  -0.502560585153 var(E) =  0.002526650079
alpha =  0.9546085671 , <E> =  -0.502393901983 var(E) =  0.00216063108683
End result: alpha =  0.961788697653 , <E> =  -0.502393901983 var(E) =  0.00216063108683
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
>>> meanEn = 0
>>> diffmeanEn = 10
>>> varE = 10
>>> i = 0
...
>>> while diffmeanEn > 0.0001 and i < numbalpha:
...     X = np.random.uniform(-2,2,(2*N,3))
...     Energy = np.zeros(shape=(steps,N))
...     lnpsi = np.zeros(shape=(steps,N))
...     Energy, lnpsi = simulate_helium_vector_min(alpha,steps,X,d)
...     meanEnNew = np.mean(Energy[4000:,:])
...     varE = np.var(Energy[4000:,:])
...
...     diffmeanEn = np.absolute(meanEnNew - meanEn)
...     meanEn = meanEnNew
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

## Hydrogen Molecule

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
```

```python
>>> X = np.random.uniform(-2,2,(2*N,3))
>>> Energy = simulate_hydrogen_atom(s,a,beta,steps,X)
>>> meanEn = np.mean(Energy[4000:,:])
>>> varE = np.var(Energy[4000:,:])
...
>>> print("beta = ",beta,", <E> = ", meanEn, "var(E) = ", varE)
```

```python
>>> def normalize(vec):
...     absvec = np.linalg.norm(vec, axis=0)
...     return absvec, vec/absvec
```

```python
>>> def apply_mask(mat, mask):
...     return mat[..., 1] * mask + mat[..., 0] * ~mask
```

```python
>>> def simulate_hydrogen_molecule(s,beta,steps,pos_walker):
...     a = fsolve(f,0.1)
...     for j in range(steps):
...         #Variational Monte Carlo step
...         pos_walker[...,1] = pos_walker[...,0] + (np.random.rand(N,3,2) - 0.5)*d
...         offset_array = np.append(s/2*np.ones([N,1,2,2]),np.zeros([N,2,2,2]), axis=1)
...         left_right_array = np.append(pos_walker + offset_array, pos_walker - offset_array, axis=2)
...         phi_1L, phi_2L, phi_1R, phi_2R = np.transpose(np.exp(np.linalg.norm(left_right_array,axis=1)/-a), axes=[1, 0, 2])
...         phi_1 = phi_1L + phi_1R
...         phi_2 = phi_2L + phi_2R
...         r_12 = -np.diff(pos_walker, axis=2)
...         r_12_abs = np.linalg.norm(r_12, axis=1)
...         psi_jastrow = np.squeeze(np.exp(r_12_abs/(2*(1+beta*r_12_abs))))
...         psi = phi_1*phi_2*psi_jastrow
...         p = (psi[:,1]/psi[:,0]) ** 2
...
...         #Create masks for different quantities going through
...         mask = p > np.random.rand(N)
...         mask_walker = np.tile(mask,(2,3,1)).T
...         mask_left_right = np.tile(mask,(4,3,1)).T
...         mask_r_abs = np.tile(mask,(1,1)).T
...         mask_r_12 = np.tile(mask,(1,3,1)).T
...
...
...         #Create accepted quantities for energy calculation
...         pos_walker[...,0] = apply_mask(pos_walker, mask_walker)
...         r_1L, r_2L, r_1R, r_2R = apply_mask(left_right_array, mask_left_right).T
...         phi_1L = apply_mask(phi_1L, mask).T
...         phi_2L = apply_mask(phi_2L, mask).T
...         phi_1R = apply_mask(phi_1R, mask).T
...         phi_2R = apply_mask(phi_2R, mask).T
...         phi_1 = phi_1L + phi_1R
...         phi_2 = phi_2L + phi_2R
...
...         r_12 = np.squeeze(-np.diff(pos_walker[...,0], axis=2)).T
...         r_12_abs, r_12_hat = normalize(r_12)
...         r_12_abs = r_12_abs.T
...         r_12_hat = r_12_hat
>>> #         r_12_abs = apply_mask(r_12_abs, mask_r_abs).T
... #         r_12 = np.transpose(apply_mask(r_12, mask_r_12), axes = [1,0,2])
... #         r_12 = np.squeeze(r_12)
... #         r_12_hat = r_12/r_12_abs
...
...         #normalize position vectors
...         r_1L_abs, r_1L_hat = normalize(r_1L)
...         r_2L_abs, r_2L_hat = normalize(r_2L)
...         r_1R_abs, r_1R_hat = normalize(r_1R)
...         r_2R_abs, r_2R_hat = normalize(r_2R)
...
...         #Calculate dot product of equation 18 of handout
...         dot_1 = (phi_1L*r_1L_hat + phi_1R*r_1R_hat)/phi_1 - (phi_2L*r_2L_hat + phi_2R*r_2R_hat)/phi_2
...         dot_2 = r_12_hat/(2*a*(1 + beta*r_12_abs*r_12_abs))
...         dot_product = np.sum(dot_1*dot_2, axis=0)
...
...
...         #Energy = (-1/a**2 + (phi_1L/r_1L_abs + phi_1R/r_1R_abs)/(a*phi_1) + (phi_2L/r_2L_abs + phi_2R/r_2R_abs)/(a*phi_2) - \
...         #         1/r_1L_abs - 1/r_1R_abs - 1/r_2L_abs - 1/r_2R_abs + 1/r_12_abs - ((4*beta + 1)*r_12_abs + 4)/(4*r_12_abs*(1 + beta*r_12_abs)**4) + \
...         #        np.dot((phi_1L*r_1L_hat + phi_1R*r_1R_hat)/phi_1 - (phi_2L*r_2L_hat + phi_2R*r_2R_hat)/phi_2, r_12/(2*a*(1 + beta*r_12_abs)**2)))
...
...         Energy[j,:] = (-1/a**2 + (phi_1L/r_1L_abs + phi_1R/r_1R_abs)/(a*phi_1) + (phi_2L/r_2L_abs + phi_2R/r_2R_abs)/(a*phi_2) - \
...                  1/r_1L_abs - 1/r_1R_abs - 1/r_2L_abs - 1/r_2R_abs + 1/r_12_abs - ((4*beta + 1)*r_12_abs + 4)/(4*r_12_abs*(1 + beta*r_12_abs)**4) + \
...                 dot_product)
...     return Energy
```

---
scrolled: true
...

```python
>>> beta = [0.1]
>>> N = 400
>>> steps = 30000
>>> d = 0.3
>>> s = 0.2
...
>>> meanEn = np.zeros(shape=(len(beta),))
>>> varE = np.zeros(shape=(len(beta),))
...
>>> for i in range(len(beta)):
...     pos_walker = np.random.uniform(-2,2,(N,3,2,2))
...     Energy = np.zeros(shape=(steps,N))
...     Energy = simulate_hydrogen_molecule(s, beta[i], steps, pos_walker)
...     meanEn[i] = np.mean(Energy[7000:,:])
...     varE[i] = np.var(Energy[7000:,:])
...
...     print("beta = ",beta[i],", <E> = ", meanEn[i], "var(E) = ", varE[i])
```

```python
>>> plt.plot(Energy[7000:,1])
```

```python
>>> pos_walker = np.random.uniform(-2,2,(N,3,2,2))
...
...
>>> #Variational Monte Carlo step
... pos_walker[...,1] = pos_walker[...,0] + (np.random.rand(N,3,2) - 0.5)*d
>>> offset_array = np.append(s/2*np.ones([N,1,2,2]),np.zeros([N,2,2,2]), axis=1)
>>> left_right_array = np.append(pos_walker + offset_array, pos_walker - offset_array, axis=2)
>>> phi_1L, phi_2L, phi_1R, phi_2R = np.transpose(np.exp(np.linalg.norm(left_right_array,axis=1)/-a), axes=[1, 0, 2])
>>> phi_1 = phi_1L + phi_1R
>>> phi_2 = phi_2L + phi_2R
>>> r_12 = -np.diff(pos_walker, axis=2)
>>> r_12_abs = np.linalg.norm(r_12, axis=1)
>>> psi_jastrow = np.squeeze(np.exp(r_12_abs/(2*(1+beta*r_12_abs))))
>>> psi = phi_1*phi_2*psi_jastrow
>>> p = (psi[:,1]/psi[:,0]) ** 2
...
>>> #Create masks for different quantities going through
... mask = p > np.random.rand(N)
>>> mask_walker = np.tile(mask,(2,3,1)).T
>>> mask_left_right = np.tile(mask,(4,3,1)).T
>>> mask_r_abs = np.tile(mask,(1,1)).T
>>> mask_r_12 = np.tile(mask,(1,3,1)).T
...
...
>>> #Create accepted quantities for energy calculation
... pos_walker[...,0] = apply_mask(pos_walker, mask_walker)
>>> r_1L, r_2L, r_1R, r_2R = apply_mask(left_right_array, mask_left_right).T
>>> phi_1L = apply_mask(phi_1L, mask).T
>>> phi_2L = apply_mask(phi_2L, mask).T
>>> phi_1R = apply_mask(phi_1R, mask).T
>>> phi_2R = apply_mask(phi_2R, mask).T
>>> phi_1 = phi_1L + phi_1R
>>> phi_2 = phi_2L + phi_2R
...
>>> r_12 = np.squeeze(-np.diff(pos_walker[...,0], axis=2)).T
>>> r_12_abs, r_12_hat = normalize(r_12)
>>> r_12_abs = r_12_abs.T
>>> r_12_hat = r_12_hat
>>> # r_12_abs = apply_mask(r_12_abs, mask_r_abs).T
... # r_12 = np.transpose(apply_mask(r_12, mask_r_12), axes = [1,0,2])
... # r_12 = np.squeeze(r_12)
... # r_12_hat = r_12/r_12_abs
...
... #normalize position vectors
... r_1L_abs, r_1L_hat = normalize(r_1L)
>>> r_2L_abs, r_2L_hat = normalize(r_2L)
>>> r_1R_abs, r_1R_hat = normalize(r_1R)
>>> r_2R_abs, r_2R_hat = normalize(r_2R)
...
>>> #Calculate dot product of equation 18 of handout
... dot_1 = (phi_1L*r_1L_hat + phi_1R*r_1R_hat)/phi_1 - (phi_2L*r_2L_hat + phi_2R*r_2R_hat)/phi_2
>>> dot_2 = r_12_hat/(2*a*(1 + beta*r_12_abs*r_12_abs))
>>> dot_product = np.sum(dot_1*dot_2, axis=0)
>>> #Energy = (-1/a**2 + (phi_1L/r_1L_abs + phi_1R/r_1R_abs)/(a*phi_1) + (phi_2L/r_2L_abs + phi_2R/r_2R_abs)/(a*phi_2) - \
... #         1/r_1L_abs - 1/r_1R_abs - 1/r_2L_abs - 1/r_2R_abs + 1/r_12_abs - ((4*beta + 1)*r_12_abs + 4)/(4*r_12_abs*(1 + beta*r_12_abs)**4) + \
... #        np.dot((phi_1L*r_1L_hat + phi_1R*r_1R_hat)/phi_1 - (phi_2L*r_2L_hat + phi_2R*r_2R_hat)/phi_2, r_12/(2*a*(1 + beta*r_12_abs)**2)))
...
...
... Energy = (-1/a**2 + (phi_1L/r_1L_abs + phi_1R/r_1R_abs)/(a*phi_1) + (phi_2L/r_2L_abs + phi_2R/r_2R_abs)/(a*phi_2) - \
...          1/r_1L_abs - 1/r_1R_abs - 1/r_2L_abs - 1/r_2R_abs + 1/r_12_abs - ((4*beta[0] + 1)*r_12_abs + 4)/(4*r_12_abs*(1 + beta[0]*r_12_abs)**4) + \
...         dot_product)
...
>>> print(Energy.shape)
```
