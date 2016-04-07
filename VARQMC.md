```python
>>> import matplotlib.pyplot as plt
>>> import numpy as np
>>> from scipy.optimize import fsolve
>>> %matplotlib inline
```

```python
>>> def Error(datavector, nblocks):
...
...     # Divide the datavector in nblocks and calculate the average value for each block
...     datavector1 = datavector[0:len(datavector) - len(datavector)%nblocks,:]
...     data_block = np.reshape(datavector1,(nblocks,-1))
...     # Used to data block specific heat
...     blockmean = np.mean(data_block,axis=1)
...     blockstd = np.std(data_block,axis=1)
...     # Calculate <A> en <A^2>
...     Mean = np.mean(blockmean)
...     # Standard deviation
...     std = np.std(blockmean)/np.sqrt(nblocks)
...     return Mean, std, blockmean, blockstd
```

```python
>>> def f(a):
...     """Coulomb cusp condition analytical expression"""
...     return (1/(1 + np.exp(-s/a)) - a)
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
>>> def simulate_harmonic_min(alpha,steps,x):
...     "A variational monte carlo method for a harmonic oscilator"
...     Energy = np.zeros(shape=(steps,N))
...     lnpsi = np.zeros(shape=(steps,N))
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
>>> def simulate_hydrogen_atom_min(alpha,steps,x,N):
...     Energy = np.zeros(shape=(steps,N))
...     lnpsi = np.zeros(shape=(steps,N))
...     "a variational monte carlo method for the hydrogen atom"
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
>>> def simulate_helium_vector_min(alpha,steps,X,d,N):
...     """A variational Monte Carlo simulation for a helium atom.
...     Based on theory of ``Computational Physics'' by J.M. Thijssen, chapter 12.2 (2nd edition)
...     X is an (2N,3) matrix with particle pairs (x,y,z) position. Particles are paired with i and N+i
...     alpha is the trial variable for the trial wave function of the form exp(-alpha * r)
...     steps is the amount of steps taken by the walkers
...     Energy (steps, N) is the energy of each particle pair at timestep j
...     """
...     Energy = np.zeros(shape=(steps,N))
...     lnpsi = np.zeros(shape=(steps,N))
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
>>> def simulate_hydrogen_molecule_min(s,beta,steps,pos_walker,N):
...     a = fsolve(f,0.1)
...     Energy = np.zeros(shape=(steps,N))
...     lnpsi = np.zeros(shape=(steps,N))
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
...
...         #normalize position vectors
...         r_1L_abs, r_1L_hat = normalize(r_1L)
...         r_2L_abs, r_2L_hat = normalize(r_2L)
...         r_1R_abs, r_1R_hat = normalize(r_1R)
...         r_2R_abs, r_2R_hat = normalize(r_2R)
...
...         #Calculate dot product of equation 18 of handout
...         dot_1 = (phi_1L*r_1L_hat + phi_1R*r_1R_hat)/phi_1 - (phi_2L*r_2L_hat + phi_2R*r_2R_hat)/phi_2
...         dot_2 = r_12_hat/(2*a*(1 + beta*r_12_abs)**2)
...         dot_product = np.sum(dot_1*dot_2, axis=0)
...
...         Energy[j,:] = (-1/a**2 + (phi_1L/r_1L_abs + phi_1R/r_1R_abs)/(a*phi_1) + (phi_2L/r_2L_abs + phi_2R/r_2R_abs)/(a*phi_2) - \
...                  1/r_1L_abs - 1/r_1R_abs - 1/r_2L_abs - 1/r_2R_abs + 1/r_12_abs - ((4*beta + 1)*r_12_abs + 4)/(4*r_12_abs*(1 + beta*r_12_abs)**4) + \
...                 dot_product + 1/s)
...         lnpsi[j,:] =  -r_12_abs**2/(2*(1+beta*r_12_abs)**2)
...     return Energy, lnpsi
```

## 1D harmonic oscilator

```python
>>> numbalpha = 1
>>> alpha = 1.2
>>> beta = 0.6
>>> N = 400
>>> steps = 4000
>>> steps_final = 30000
>>> wastesteps = 4000
>>> d = 0.05 #movement size
>>> diffmeanEn = 10
>>> meanEn = 0
>>> i = 0
...
>>> while diffmeanEn > 0.0001 and i < numbalpha:
...     x = np.random.uniform(-1,1,(N))
...     Energy, lnpsi = simulate_harmonic_min(alpha,steps,x)
...     meanEnNew = np.mean(Energy)
...     varE = np.var(Energy)
...     diffmeanEn = np.absolute(meanEnNew - meanEn)
...     meanEn = meanEnNew
...     print("alpha = ",alpha,", <E> = ", meanEn, "var(E) = ", varE)
...
...     meanlnpsi = np.mean(lnpsi)
...     meanEtimeslnpsi = np.mean(lnpsi*Energy)
...     dEdalpha = 2*(meanEtimeslnpsi-meanEn*meanlnpsi)
...     #alpha -= ((i+1)**(-beta))*dEdalpha
...     alpha -= 0.8 * dEdalpha
...     i += 1
...
>>> print("End result: alpha = ",alpha,", <E> = ", meanEn, 'var(E) = ', varE)
alpha =  1.2 , <E> =  0.598634277305 var(E) =  0.637661911352
End result: alpha =  0.771319723461 , <E> =  0.598634277305 var(E) =  0.637661911352
```

```python
>>> rslt = view.pull('alpha', targets=[0,1,2,3])
>>> alpha = np.mean(rslt.get())
>>> print(alpha)
0.816064323819
```

```python
>>> %%px
... Energy_final = np.zeros(shape=(steps_final,))
... X = np.random.uniform(-2,2,N)
... Energy_final = simulate_harmonic_min(alpha,steps_final,X)[0]
```

```python
>>> #rslt = view.pull('Energy_final', targets=[0,1,2,3])
... #Energy_final = np.transpose(np.asarray(rslt.get()),axes=[1,0,2]).reshape(30000,-1)
... #print(Energy_final.shape)
... varE_final = np.var(Energy_final[4000:,:])
>>> mean_final = np.mean(Energy_final[4000:,:])
(30000, 1600)
```

```python
>>> rslt = view.pull('Energy_final', targets=1)
>>> Energy_final1 = rslt.get()
>>> print(np.array_equal(Energy_final1[:,0],Energy_final[:,400]))
True
```

## Hydrogen Atom

```python
>>> numbalpha = 300
>>> alpha = 0.1
>>> N = 400
>>> steps = 100
>>> steps_final = 30000
>>> d = 0.5 #movement size
>>> varE = 10
>>> i = 0
...
>>> alpha_old = alpha
>>> dalpha = 1
...
>>> while abs(dalpha) > 1e-6 and i < numbalpha:
...     x = np.random.uniform(-1,1,(N,3))
...     Energy = np.zeros(shape=(steps,N))
...     lnpsi = np.zeros(shape=(steps,N))
...     Energy, lnpsi = simulate_hydrogen_atom_min(alpha,steps,x,N)
...
...     meanEn = np.mean(Energy)
...     varE = np.var(Energy)
...
...     #print("alpha = ",alpha,", <E> = ", meanEn, "dalpha(%) = ", dalpha)
...
...     meanlnpsi = np.mean(lnpsi)
...     meanEtimeslnpsi = np.mean(lnpsi*Energy)
...     dEdalpha = 2*(meanEtimeslnpsi-meanEn*meanlnpsi)
...     alpha -= 0.3*dEdalpha
...     dalpha = (alpha-alpha_old)/alpha_old
...     i += 1
...     alpha_old = alpha
...
>>> print("End result: alpha = ",alpha, 'iteration # = ', i)
...
>>> Energy_final = np.zeros(shape=(steps_final,N))
>>> x = np.random.uniform(-1,1,(N,3))
>>> Energy_final = simulate_hydrogen_atom_min(alpha,steps_final,x,N)[0]
>>> Energy_truncated = Energy_final[4000:,:]
>>> print(Energy_truncated.shape)
>>> meanEn = Error(Energy_truncated,4)[0]
>>> stdE = Error(Energy_truncated,4)[1]
...
>>> print("<E> = ", meanEn, "with error: ", stdE)
```

## Helium Atom

```python
>>> numbalpha = 200
>>> #alpha = 0.132
... alpha = 0.5
>>> beta = 0.6
>>> N = 400
>>> steps = 4000
>>> steps_final = 30000
>>> d = 0.3 #movement size
...
>>> i = 0
>>> alpha_old = alpha
>>> dalpha = 1
...
>>> while abs(dalpha) > 1e-4 and i < numbalpha:
...     X = np.random.uniform(-2,2,(2*N,3))
...     Energy = np.zeros(shape=(steps,N))
...     lnpsi = np.zeros(shape=(steps,N))
...     Energy, lnpsi = simulate_helium_vector_min(alpha,steps,X,d,N)
...     #meanEnNew = np.mean(Energy)
...     varE = np.var(Energy)
...     meanEn = np.mean(Energy)
...     #diffmeanEn = np.absolute(meanEnNew - meanEn)
...     #meanEn = meanEnNew
...
...
...     print("alpha = ",alpha,", <E> = ", meanEn, ", dalpha(%) = ", dalpha*100)
...
...     meanlnpsi = np.mean(lnpsi)
...     meanEtimeslnpsi = np.mean(lnpsi*Energy)
...     dEdalpha = 2*(meanEtimeslnpsi-meanEn*meanlnpsi)
...     alpha -=  0.03*dEdalpha
...     dalpha = (alpha-alpha_old)/alpha_old
...     i += 1
...     alpha_old = alpha
...
>>> print("End result: alpha = ",alpha, "in ", i, "iterations")
...
>>> Energy_final = np.zeros(shape=(steps_final,))
>>> X = np.random.uniform(-2,2,(2*N,3))
>>> Energy_final = simulate_helium_vector_min(alpha,steps_final,X,d,N)[0]
>>> varE_final = np.var(Energy_final[4000:,:])
>>> mean_final = np.mean(Energy_final[4000:,:])
...
>>> print("Final Energy at alpha(",alpha,") =", mean_final, ", var(E) = ", varE_final)
```

## Hydrogen Molecule

```python
>>> numbbeta = 10000
>>> beta = 0.55
>>> zeta = 0.51
>>> N = 400
>>> steps = 1000
>>> d = 2.0
>>> s_row = [1.4011]
>>> nblocks = 10
...
>>> beta_old = beta
>>> dbeta = 1
>>> i = 0
...
>>> for i in range(len(s_row)):
...     s = s_row[i]
...     while abs(dbeta) > 1e-4 and i < numbbeta:
...         pos_walker = np.random.uniform(-2,2,(N,3,2,2))
...         Energy = np.zeros(shape=(steps,N))
...         lnpsi = np.zeros(shape=(steps,N))
...
...         Energy, lnpsi = simulate_hydrogen_molecule_min(s, beta, steps, pos_walker,N)
...         varE = np.var(Energy)
...         meanEn = np.mean(Energy)
...
...         #print("beta = ",beta,", <E> = ", meanEn, "iteration = ", i, "dbeta(%) = ", dbeta*100)
...
...         meanlnpsi = np.mean(lnpsi)
...         meanEtimeslnpsi = np.mean(lnpsi*Energy)
...         dEdbeta = 2*(meanEtimeslnpsi-meanEn*meanlnpsi)
...         beta -= 0.5*dEdbeta
...         dbeta = (beta - beta_old)/beta_old
...         beta_old = beta
...         i += 1
...
...     #print("End result: beta = ",beta," in ", i,"iterations.")
...
...     Energy_final = np.zeros(shape=(steps_final,))
...     pos_walker = np.random.uniform(-2,2,(N,3,2,2))
...     Energy_final = simulate_hydrogen_molecule_min(s, beta, steps_final, pos_walker, N)[0]
...
...     Energy_truncated = Energy_final[7000:,:]
...     varE_final = np.var(Energy_truncated)
...     mean_error_calculated = Error(Energy_truncated,nblocks)[0]
...     std_error_calculated = Error(Energy_truncated,nblocks)[1]
...
...     print("mean with error function: ", mean_error_calculated, "and error: ", std_error_calculated)
mean with error function:  -1.15094264375 and error:  0.000230339655967
```

```python
>>> rslt = view.pull('beta', targets = c.ids)
>>> beta = np.mean(rslt.get())
>>> print(beta)
0.578855402233
```

```python
>>> %%px
... N = 100
... steps_final = 300000
...
... Energy_final = np.zeros(shape=(steps_final,))
... pos_walker = np.random.uniform(-2,2,(N,3,2,2))
... Energy_final = simulate_hydrogen_molecule_min(s, beta, steps_final, pos_walker, N)[0]
```

```python
>>> rslt = view.pull('Energy_final', targets=c.ids)
>>> Energy_final = np.transpose(np.asarray(rslt.get()),axes=[1,0,2]).reshape(300000,-1)
>>> print(Energy_final.shape)
...
>>> Energy_truncated = Energy_final[7000:,:]
>>> varE_final = np.var(Energy_truncated)
>>> mean_error_calculated = Error(Energy_truncated,nblocks)[0]
>>> std_error_calculated = Error(Energy_truncated,nblocks)[1]
...
...
>>> print("mean with error function: ", mean_error_calculated, "and error: ", std_error_calculated)
(300000, 400)
mean with error function:  -1.15092518475 and error:  5.7517652745e-05
```

```python

```
