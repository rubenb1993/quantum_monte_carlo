```python
>>> # install ipyparallel
... # ipcluster start -n 4       to start your engines!
```

```python
>>> #defining parallel clients
... import ipyparallel as ipp
>>> c = ipp.Client()
>>> view = c[0:3]
>>> print(c.ids)
[0, 1, 2, 3]
```

```python
>>> %%px
... #import libraries on parallel engines
... import numpy as np
... from scipy.optimize import fsolve
```

```python
>>> import matplotlib.pyplot as plt
>>> import numpy as np
>>> from scipy.optimize import fsolve
>>> %matplotlib inline
```

```python
>>> def Error(datavector, nblocks):
...     """
...     Mean and error calculation of a quantum monte carlo proces.
...     Datavector should be a (steps, N) matrix. Returns a scalar value for the mean of this vector and a standard error
...     determined by creating n blocks and calculating the standard deviation. """
...     # Divide the datavector in nblocks and calculate the average value for each block
...     datavector1 = datavector[0:len(datavector) - len(datavector)%nblocks,:]
...     data_block = np.reshape(datavector1,(-1,N,nblocks))
...     # Used to data block specific heat
...     blockmean = np.mean(data_block,axis=(0,1))
...     blockstd = np.std(data_block,axis=(0,1))
...     # Calculate <A> en <A^2>
...     Mean = np.mean(blockmean)
...     # Standard deviation
...     std = np.std(blockmean)/np.sqrt(nblocks)
...     return Mean, std
```

```python
>>> def f(a):
...     """Coulomb cusp condition analytical expression"""
...     return (1/(1 + np.exp(-s/a)) - a)
...
>>> def normalize(vec):
...     """Normalizes a vector of (3,N), where there are 3 dimensions and N walkers. Returns a (N,) vector with the length
...     of the N vectors and a (3,N) normalized vector"""
...     absvec = np.linalg.norm(vec, axis=0)
...     return absvec, vec/absvec
...
>>> def apply_mask(mat, mask):
...     """Applies the mask created by the Variational monte carlo proces. The matrix should contain 1 more dimension than
...     the mask, and the matrix has the old values in [...,0] and the new shifted values in [...,1]. The mask should have
...     the same dimensions and the [...] part of the matrix, and should contain either true or false values."""
...     return mat[..., 1] * mask + mat[..., 0] * ~mask
```

```python
>>> %%px
... def f(a):
...     """Coulomb cusp condition analytical expression"""
...     return (1/(1 + np.exp(-s/a)) - a)
...
... def normalize(vec):
...     """Normalizes a vector of (3,N), where there are 3 dimensions and N walkers. Returns a (N,) vector with the length
...     of the N vectors and a (3,N) normalized vector"""
...     absvec = np.linalg.norm(vec, axis=0)
...     return absvec, vec/absvec
...
... def apply_mask(mat, mask):
...     """Applies the mask created by the Variational monte carlo proces. The matrix should contain 1 more dimension than
...     the mask, and the matrix has the old values in [...,0] and the new shifted values in [...,1]. The mask should have
...     the same dimensions and the [...] part of the matrix, and should contain either true or false values."""
...     return mat[..., 1] * mask + mat[..., 0] * ~mask
```

```python
>>> def simulate_hydrogen_molecule(s,beta,steps,N):
...     """A variational Monte Carlo simulation for a hydrogen molecule.
...     Based on theory of ``Computational Physics'' by J.M. Thijssen, chapter 12 (2nd edition)
...     s is the internuclear distance between the 2 protons
...     beta is a trial scalar value in order to determine the trial wavefunction
...     steps is a scalar for the amount of timesteps taken in the VQMC simulation.
...     pos_walker is a (N,3,2,2) matrix, containing the information of the (x,y,z) position of N electron pairs (first 3 dimensions),
...         in the old and new position (4th dimension).
...     N is the amount of electron pairs, and with that walkers.
...     Energy (steps, N) is the energy of each particle pair at timestep j.
...     lnpsi (steps,N) is a matrix used to determine the minimum energy.
...     """
...     pos_walker = np.random.uniform(-2,2,(N,3,2,2))
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

```python
>>> %%px
... def simulate_hydrogen_molecule_min(s,beta,steps,pos_walker,N):
...     """A variational Monte Carlo simulation for a hydrogen molecule.
...     Based on theory of ``Computational Physics'' by J.M. Thijssen, chapter 12 (2nd edition)
...     s is the internuclear distance between the 2 protons
...     beta is a trial scalar value in order to determine the trial wavefunction
...     steps is a scalar for the amount of timesteps taken in the VQMC simulation.
...     pos_walker is a (N,3,2,2) matrix, containing the information of the (x,y,z) position of N electron pairs (first 3 dimensions),
...         in the old and new position (4th dimension).
...     N is the amount of electron pairs, and with that walkers.
...     Energy (steps, N) is the energy of each particle pair at timestep j.
...     lnpsi (steps,N) is a matrix used to determine the minimum energy.
...     """
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

## Hydrogen Molecule

---
scrolled: true
...

```python
>>> numbbeta = 150
>>> steps = 1000
>>> gamma = 0.5
>>> steps_final = 8000
>>> d = 2.0
>>> s_row = np.append(np.linspace(1,2,11),1.4011)
>>> error_blocks = 10
>>> N = 400
>>> wastesteps = 70
...
>>> energy_graph_data = np.zeros(shape=(len(s_row),3))
...
>>> for j in range(len(s_row)):
...     s = s_row[j]
...     beta = 0.55
...     beta_old = beta
...     dbeta = 1
...     i = 0
...     #Determine best beta with a steepest descent method
...     while abs(dbeta) > 5e-5 and i < numbbeta:
...         #Do VQMC step with old beta
...         Energy = np.zeros(shape=(steps,N))
...         lnpsi = np.zeros(shape=(steps,N))
...         Energy, lnpsi = simulate_hydrogen_molecule(s_row[j], beta, steps,N)
...         varE = np.var(Energy)
...         meanEn = np.mean(Energy)
...
...         #print("beta = ",beta,", <E> = ", meanEn, "iteration = ", i, "dbeta(%) = ", dbeta*100)
...
...         #Determine new beta
...         meanlnpsi = np.mean(lnpsi)
...         meanEtimeslnpsi = np.mean(lnpsi*Energy)
...         dEdbeta = 2*(meanEtimeslnpsi-meanEn*meanlnpsi)
...         beta -= gamma*dEdbeta
...         dbeta = (beta - beta_old)/beta_old
...         beta_old = beta
...         i += 1
...
...     #give necessary parameters to the engines. This increases the amount of walkers 4-fold
...     view.push(dict(beta = beta, s = s, j = j, d = d, N = int(N/len(c.ids)), steps_final = steps_final), targets = c.ids)
...
...     #print("End result: beta = ",beta," in ", i,"iterations.")
...
...     #Parallel computing the energy
...     %px Energy_hydrogen_mol = np.zeros(shape=(steps_final,))
...     %px Energy_hydrogen_mol = simulate_hydrogen_molecule(s, beta, steps_final, N)[0]
...     #End Parallel computing
...
...     #Gather and reshape the energy from the parallel engines
...     rslt = view.pull('Energy_hydrogen_mol', targets=c.ids)
...     Energy_final = np.transpose(np.asarray(rslt.get()),axes=[1,0,2]).reshape(steps_final,-1)
...
...     #Calculate final Energy using the error function
...     Energy_truncated = Energy_final[wastesteps:,:]
...     varE_final = np.var(Energy_truncated)
...     hydrogen_mol_energy, hydrogen_mol_error = Error(Energy_truncated,error_blocks)
...
...     #Save data every timestep in order to not lose intermediate data when stopped.
...     energy_graph_data[j,0] = hydrogen_mol_energy
...     energy_graph_data[j,1] = hydrogen_mol_error
...     energy_graph_data[j,2] = beta
...     np.save('20160408_energy_graph_data', energy_graph_data)
...     print("Done with step", j+1, "out of ",len(s_row))
...     #print("mean with error function: ", mean_energy, "and error: ", std_error)
...
>>> straks = time.time()
>>> print("Time elapsed: ", straks-nu, "s")
```

```python
>>> nu = time.time()
>>> numbbeta = 150
>>> steps = 800000
>>> d = 2.0
>>> beta_row = np.linspace(0.1,1,10)
>>> s = 1.4011 #at approximate hydrogen molecule distance
>>> error_blocks = 10
>>> N = 400
>>> wastesteps = 7000
...
>>> beta_graph_data = np.zeros(shape=(len(beta_row),3))
...
>>> for j in range(len(beta_row)):
...     beta = beta_row[j]
...
...     #give necessary parameters to the engines. This increases the amount of walkers 4-fold
...     view.push(dict(beta = beta, s = s, j = j, d = d, N = int(N/len(c.ids)), steps = steps), targets = c.ids)
...
...     #print("End result: beta = ",beta," in ", i,"iterations.")
...
...     #Parallel computing the energy
...     %px Energy_final = np.zeros(shape=(steps,))
...     %px pos_walker = np.random.uniform(-2,2,(N,3,2,2))
...     %px Energy_final = simulate_hydrogen_molecule_min(s, beta, steps, pos_walker, N)[0]
...     #End Parallel computing
...
...     #Gather and reshape the energy from the parallel engines
...     rslt = view.pull('Energy_final', targets=c.ids)
...     Energy_final = np.transpose(np.asarray(rslt.get()),axes=[1,0,2]).reshape(steps,-1)
...
...     #Calculate final Energy using the error function
...     Energy_truncated = Energy_final[wastesteps:,:]
...     varE_final = np.var(Energy_truncated)
...     energy_beta_hydrogen, error_beta_hydrogen = Error(Energy_truncated,error_blocks)
...
...     beta_graph_data[j,0] = energy_beta_hydrogen
...     beta_graph_data[j,1] = error_beta_hydrogen
...     beta_graph_data[j,2] = beta
...     np.save('beta_graph_data', beta_graph_data) #save for every value of beta to save intermediate results
...     print("Done with step ", j+1, "out of ", len(beta_row))
...     #print("mean with error function: ", mean_energy, "and error: ", std_error)
...
>>> straks = time.time()
>>> print("Time elapsed: ", straks-nu, "s")
Done with step  0 out of  10
Done with step  1 out of  10
Done with step  2 out of  10
Done with step  3 out of  10
Done with step  4 out of  10
Done with step  5 out of  10
Done with step  6 out of  10
Done with step  7 out of  10
Done with step  8 out of  10
Done with step  9 out of  10
Time elapsed:  7699.671926021576 s
```
