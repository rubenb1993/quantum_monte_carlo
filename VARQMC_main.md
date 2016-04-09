```python
>>> import matplotlib.pyplot as plt
>>> import numpy as np
>>> from scipy.optimize import fsolve
>>> from matplotlib import rc
...
>>> # Define font for figures
... rc('font', **{'family': 'serif', 'serif': ['Computer Modern']})
>>> rc('text', usetex=True)
...
>>> %matplotlib inline
```

```python
>>> def Error(datavector, nblocks):
...     """
...     Mean and error calculation of a quantum monte carlo proces.
...     Datavector should be a (steps, N) matrix. Returns a scalar value for the mean of this vector and a standard error
...     determined by creating n blocks and calculating the standard deviation. """
...     # Divide the datavector in nblocks and calculate the average value for each block
...     datavector1 = datavector[0:len(datavector) - len(datavector)%nblocks, :]
...     data_block = np.reshape(datavector1, (-1,N,nblocks))
...     blockmean = np.mean(data_block, axis=(0,1))
...     mean = np.mean(blockmean)
...     stderr = np.std(blockmean)/np.sqrt(nblocks)
...     return mean, stderr
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
>>> def simulate_harmonic_oscillator(alpha, steps, N):
...     """A variational monte carlo method for a harmonic oscilator.
...     Requires a scalar value for alpha used in the wavefunction, the amount of steps to be taken, and an initial position
...     vector x (N,) big where N is the amount of walkers."""
...     x = np.random.uniform(-1, 1, (N))
...     Energy = np.zeros(shape=(steps, N))
...     lnpsi = np.zeros(shape=(steps, N))
...     for j in range(steps):
...         x_new = x + (np.random.rand(len(x)) - 0.5)*d
...         p = (np.exp(-alpha*x_new*x_new) / np.exp(-alpha*x*x))**2 #calculate ratio of psi^2 for old and new
...         m = p > np.random.rand(len(x))
...         x = apply_mask(x,m)
...         Energy[j,:] = alpha + x*x *(0.5 - 2*(alpha*alpha))
...         lnpsi[j,:] = -x*x #determine the logarithm of the wavefunction for energy minimalization
...     return Energy, lnpsi
```

```python
>>> def simulate_hydrogen_atom(alpha, steps, N):
...     x = np.random.uniform(-1, 1, (N, 3))
...     Energy = np.zeros(shape=(steps, N))
...     lnpsi = np.zeros(shape=(steps, N))
...     """a variational monte carlo method for the hydrogen atom.
...     Requires a scalar value for alpha used in the wavefunction, the amount of steps to be taken, an initial position
...     vector x (N,) and the amount of walkers N."""
...     for j in range(steps):
...         x_new = x + (np.random.random_sample(np.shape(x)) - 0.5)*d
...         p = (np.exp(-alpha*np.linalg.norm(x_new, axis=1)) / np.exp(-alpha*np.linalg.norm(x, axis=1)))**2
...         m = (p > np.random.rand(N)).reshape(-1, 1)
...         x = x_new*m + x*~m
...         Energy[j,:] = -1/np.linalg.norm(x, axis=1) -alpha/2*(alpha - 2/np.linalg.norm(x, axis=1))
...         lnpsi[j,:] = -np.linalg.norm(x) #determine the logarithm of the wave function to minimize energy
...     return Energy, lnpsi
```

```python
>>> def simulate_helium_atom(alpha, steps, d, N):
...     """A variational Monte Carlo simulation for a helium atom.
...     Based on theory of ``Computational Physics'' by J.M. Thijssen, chapter 12.2 (2nd edition)
...     X is an (2N,3) matrix with particle pairs (x,y,z) position. Particles are paired with i and N+i
...     alpha is the trial variable for the trial wave function of the form exp(-alpha * r)
...     steps is the amount of steps taken by the walkers
...     Energy (steps, N) is the energy of each particle pair at timestep j
...     lnpsi (steps,N) is a scalar used to determine the minimum energy of the wavefunction.
...     """
...     X = np.random.uniform(-2, 2, (2*N,3))
...     Energy = np.zeros(shape=(steps, N))
...     lnpsi = np.zeros(shape=(steps, N))
...     for j in range(steps):
...         X_new = X + (np.random.rand(2*N, 3) - 0.5) * d
...         r_old = np.linalg.norm(X,axis=1)
...         r_new = np.linalg.norm(X_new,axis=1)
...         r12_old = np.linalg.norm(X[0:N, :] - X[N:, :],axis=1)
...         r12_new = np.linalg.norm(X_new[0:N, :] - X_new[N:, :],axis=1)
...
...         psi_fact_old = 1 + alpha * r12_old
...         psi_fact_new = 1 + alpha * r12_new
...
...         psi_old = np.exp(-2*r_old[0:N] - 2*r_old[N:]) * np.exp(r12_old/(2*psi_fact_old))
...         psi_new= np.exp(-2*r_new[0:N] - 2*r_new[N:]) * np.exp(r12_new/(2*psi_fact_new))
...
...         p = (psi_new/psi_old)**2
...         m = p>np.random.rand(N)
...         m = np.transpose(np.tile(m, (3, 2))) #make from m a 400,3 matrix by repeating m
...         X = X_new*(m) + X*~(m)
...
...         r1r2_diff = X[0:N,:] - X[N:,:]
...         r1_length = np.tile(np.linalg.norm(X[0:N,:], axis=1),(3, 1)).T #200,3 length vector to normalize r1
...         r2_length = np.tile(np.linalg.norm(X[N:, :],axis=1),(3, 1)).T #200,3 length vector to normalize r2
...         r1r2_diff_hat = X[0:N, :]/r1_length - X[N:, :]/r2_length
...
...         r12 = np.linalg.norm(X[0:N, :] - X[N:, :],axis=1)
...         psi_fact = 1 + alpha*r12
...
...         dot_product = np.sum(r1r2_diff_hat * r1r2_diff,axis=1)
...
...         Energy[j,:] = -4 + dot_product/(r12*psi_fact**2) - 1/(r12*psi_fact**3) - 1/(4*psi_fact**4) + 1/r12
...         lnpsi[j,:] = -2*(r12*r12)/(4*(1+alpha*r12)*(1+alpha*r12))
...     return Energy, lnpsi
```

```python
>>> def simulate_hydrogen_molecule(s, beta, steps, N):
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
...     pos_walker = np.random.uniform(-2, 2, (N,3,2,2))
...     a = fsolve(f, 0.1)
...     Energy = np.zeros(shape=(steps, N))
...     lnpsi = np.zeros(shape=(steps, N))
...     for j in range(steps):
...         #Variational Monte Carlo step
...         pos_walker[..., 1] = pos_walker[..., 0] + (np.random.rand(N, 3, 2) - 0.5)*d
...         offset_array = np.append(s/2*np.ones([N, 1, 2, 2]),np.zeros([N, 2, 2, 2]), axis=1)
...         left_right_array = np.append(pos_walker + offset_array, pos_walker - offset_array, axis=2)
...         phi_1L, phi_2L, phi_1R, phi_2R = np.transpose(np.exp(np.linalg.norm(left_right_array, axis=1)/-a), axes=[1, 0, 2])
...         phi_1 = phi_1L + phi_1R
...         phi_2 = phi_2L + phi_2R
...         r_12 = -np.diff(pos_walker, axis=2)
...         r_12_abs = np.linalg.norm(r_12, axis=1)
...         psi_jastrow = np.squeeze(np.exp(r_12_abs/(2*(1 + beta*r_12_abs))))
...         psi = phi_1*phi_2*psi_jastrow
...         p = (psi[:, 1]/psi[:, 0]) ** 2
...
...         #Create masks for different quantities going through
...         mask = p > np.random.rand(N)
...         mask_walker = np.tile(mask,(2, 3, 1)).T
...         mask_left_right = np.tile(mask,(4, 3, 1)).T
...         mask_r_abs = np.tile(mask,(1, 1)).T
...         mask_r_12 = np.tile(mask,(1, 3, 1)).T
...
...
...         #Create accepted quantities for energy calculation
...         pos_walker[..., 0] = apply_mask(pos_walker, mask_walker)
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
...         Energy[j, :] = (-1/a**2 + (phi_1L/r_1L_abs + phi_1R/r_1R_abs)/(a*phi_1) + (phi_2L/r_2L_abs + phi_2R/r_2R_abs)/(a*phi_2) - \
...                  1/r_1L_abs - 1/r_1R_abs - 1/r_2L_abs - 1/r_2R_abs + 1/r_12_abs - ((4*beta + 1)*r_12_abs + 4)/(4*r_12_abs*(1 + beta*r_12_abs)**4) + \
...                 dot_product + 1/s)
...         lnpsi[j,:] =  -r_12_abs**2/(2*(1 + beta*r_12_abs)**2)
...     return Energy, lnpsi
```

## 1D harmonic oscilator

```python
>>> numbalpha = 100
>>> alpha = 1.2 #starting guess for alpha
>>> gamma = 0.8 #steepest descent parameter
>>> N = 400
>>> steps = 4000
>>> steps_final = 30000
>>> wastesteps = 4000
>>> d = 0.05 #movement size
>>> i = 0
>>> error_blocks = 5
...
>>> dalpha = 1
>>> alpha_old = alpha
...
>>> #optimize alpha with a steepest descent method
... while abs(dalpha) > 5e-5 and i < numbalpha:
...     #Do VQMC step with old value for alpha
...     Energy, lnpsi = simulate_harmonic_oscillator(alpha, steps, N)
...     meanEn = np.mean(Energy)
...     varE = np.var(Energy)
...     #print("alpha = ",alpha,", <E> = ", meanEn, "var(E) = ", varE)
...
...     #determine new alpha
...     meanlnpsi = np.mean(lnpsi)
...     meanEtimeslnpsi = np.mean(lnpsi*Energy)
...     dEdalpha = 2*(meanEtimeslnpsi-meanEn*meanlnpsi)
...     alpha -= gamma * dEdalpha
...     dalpha = (alpha-alpha_old)/alpha_old
...     alpha_old = alpha
...     i += 1
...
>>> #print("End result: alpha = ",alpha,", <E> = ", meanEn, 'var(E) = ', varE)
... #Determine the final energy with the optimized value for alpha
... Energy_final = np.zeros(shape=(steps_final,))
>>> Energy_final = simulate_harmonic_oscillator(alpha, steps_final, N)[0]
...
>>> Energy_harmonic, error_harmonic = Error(Energy_final, error_blocks)
>>> print("<E> = ", Energy_harmonic, "with error ", error_harmonic)
<E> =  0.756195031545 with error  7.88867064184e-08
```

## Hydrogen Atom

```python
>>> numbalpha = 30
>>> alpha = 0.1
>>> N = 400
>>> steps = 100
>>> steps_final = 30000
>>> d = 0.5 #movement size
>>> equilibrium_steps = 4000
>>> error_blocks = 5
...
>>> alpha_old = alpha
>>> dalpha = 1
>>> i = 0
...
>>> #Approximate the optimal value of alpha using a steepest descent method
... while abs(dalpha) > 1e-6 and i < numbalpha:
...     #Do VQMC steps for old value of alpha
...     Energy = np.zeros(shape=(steps, N))
...     lnpsi = np.zeros(shape=(steps, N))
...     Energy, lnpsi = simulate_hydrogen_atom(alpha, steps, N)
...     meanEn = np.mean(Energy)
...     varE = np.var(Energy)
...
...     #Determine new value for alpha
...     meanlnpsi = np.mean(lnpsi)
...     meanEtimeslnpsi = np.mean(lnpsi*Energy)
...     dEdalpha = 2*(meanEtimeslnpsi-meanEn*meanlnpsi)
...     alpha -= 0.3*dEdalpha
...     dalpha = (alpha - alpha_old)/alpha_old
...     i += 1
...     alpha_old = alpha
...
>>> print("End result: alpha = ",alpha, 'iteration # = ', i)
...
>>> #Determine the final energy with the approximated value of alpha
... Energy_final = np.zeros(shape=(steps_final, N))
>>> Energy_final = simulate_hydrogen_atom(alpha, steps_final, N)[0]
>>> Energy_truncated = Energy_final[equilibrium_steps:,:]
>>> mean_hydrogen_atom, error_hydrogen_atom  = Error(Energy_truncated, error_blocks)
...
>>> print("<E> = ", mean_hydrogen_atom, "with error: ", error_hydrogen_atom)
```

## Helium Atom

```python
>>> numbalpha = 20
>>> alpha = 0.5
>>> gamma = 0.03
>>> beta = 0.6
>>> N = 400
>>> steps = 4000
>>> steps_final = 30000
>>> d = 0.3 #movement size
>>> wastesteps = 4000
>>> error_blocks = 5
...
>>> i = 0
>>> alpha_old = alpha
>>> dalpha = 1
...
>>> #Approximate the optimal value of alpha with a steepest descent method
... while abs(dalpha) > 1e-4 and i < numbalpha:
...     #Do a VQMC step with old alpha
...     Energy = np.zeros(shape=(steps, N))
...     lnpsi = np.zeros(shape=(steps, N))
...     Energy, lnpsi = simulate_helium_atom(alpha, steps, d, N)
...     varE = np.var(Energy)
...     meanEn = np.mean(Energy)
...
...     #Determine new value of alpha
...     meanlnpsi = np.mean(lnpsi)
...     meanEtimeslnpsi = np.mean(lnpsi*Energy)
...     dEdalpha = 2*(meanEtimeslnpsi-meanEn*meanlnpsi)
...     alpha -=  gamma*dEdalpha
...     dalpha = (alpha-alpha_old)/alpha_old
...     i += 1
...     alpha_old = alpha
...
>>> print("End result: alpha = ",alpha, "in ", i, "iterations")
...
>>> #Determine the energy with the optimized value of alpha
... Energy_helium = np.zeros(shape=(steps_final, ))
>>> Energy_helium = simulate_helium_atom(alpha,steps_final, d, N)[0]
>>> Energy_helium_truncated = Energy_helium[wastesteps:, :]
>>> helium_energy, helium_error = Error(Energy_helium_truncated, error_blocks)
>>> helium_variance = np.var(Energy_helium_truncated)
...
>>> print("Final Energy at alpha(",alpha,") =", helium_energy, ", with error = ", helium_error, "and variance ", helium_variance)
End result: alpha =  0.449507948461 in  20 iterations
Final Energy at alpha( 0.449507948461 ) = -2.85893823217 , with error =  0.000708575900182 and variance  0.0912794879072
```

## Hydrogen Molecule

```python
>>> numbbeta = 5
>>> steps = 1000
>>> gamma = 0.5
>>> steps_final = 8000
>>> d = 2.0
>>> s_row = [1.4, 1.4011, 1.5]
>>> error_blocks = 39
>>> N = 400
>>> wastesteps = 20000
...
>>> energy_graph_data = np.zeros(shape=(len(s_row), 3))
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
...         pos_walker = np.random.uniform(-2, 2, (N, 3, 2, 2))
...         Energy = np.zeros(shape=(steps,N))
...         lnpsi = np.zeros(shape=(steps,N))
...         Energy, lnpsi = simulate_hydrogen_molecule_min(s_row[j], beta, steps, pos_walker,N)
...         varE = np.var(Energy)
...         meanEn = np.mean(Energy)
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
...
...     Energy_final = np.zeros(shape=(steps_final, ))
...     pos_walker = np.random.uniform(-2, 2,(N, 3, 2, 2))
...     Energy_final = simulate_hydrogen_molecule_min(s, beta, steps_final, pos_walker, N)[0]
...
...
...     #Calculate final Energy using the error function
...     Energy_truncated = Energy_final[wastesteps:, :]
...     varE_final = np.var(Energy_truncated)
...     hydrogen_mol_energy, hydrogen_mol_error = Error(Energy_truncated, error_blocks)
...
...     #Save data every timestep in order to not lose intermediate data when stopped.
...     energy_graph_data[j,0] = hydrogen_mol_energy
...     energy_graph_data[j,1] = hydrogen_mol_error
...     energy_graph_data[j,2] = beta
...     np.save('20160409_energy_long_corr_time_s_1_440115', energy_graph_data)
...     print("Done with step", j+1, "out of ",len(s_row))
...     #print("mean with error function: ", mean_energy, "and error: ", std_error)
...
>>> straks = time.time()
>>> print("Time elapsed: ", straks-nu, "s")
Done with step 1 out of  3
Done with step 2 out of  3
Done with step 3 out of  3
Time elapsed:  2478.7046654224396 s
```

```python
>>> numbbeta = 150
>>> steps = 800000
>>> d = 2.0
>>> beta_row = np.linspace(0.1, 1, 10)
>>> s = 1.4011 #at approximate hydrogen molecule distance
>>> error_blocks = 10
>>> N = 400
>>> wastesteps = 7000
...
>>> beta_graph_data = np.zeros(shape=(len(beta_row), 3))
...
>>> for j in range(len(beta_row)):
...     beta = beta_row[j]
...     Energy_final = np.zeros(shape=(steps, ))
...     Energy_final = simulate_hydrogen_molecule_min(s, beta, steps, N)[0]
...
...     #Calculate final Energy using the error function
...     Energy_truncated = Energy_final[wastesteps:, :]
...     varE_final = np.var(Energy_truncated)
...     energy_beta_hydrogen, error_beta_hydrogen = Error(Energy_truncated, error_blocks)
...
...     beta_graph_data[j,0] = energy_beta_hydrogen
...     beta_graph_data[j,1] = error_beta_hydrogen
...     beta_graph_data[j,2] = beta
...     np.save('20160409_beta_graph_data', beta_graph_data) #save for every value of beta to save intermediate results
```

```python
>>> fig = plt.figure(figsize=(3,1.85))
...
>>> s_row = np.append(np.linspace(1, 2, 11), 1.4011)
>>> s_row_finer = np.linspace(0.15, 0.95, 9)
>>> loc_en_1 = np.load('20160408_energy_graph_data.npy')
>>> loc_en_2 = np.load('20160409_energy_graph_data.npy')
...
>>> sort = s_row.argsort()
>>> x_axis = s_row[sort]
>>> y_axis = loc_en_1[:, 0][sort]
...
>>> ax2 = fig.add_subplot(111)
>>> s1_energy = ax2.plot(x_axis, y_axis, 'ok-', markersize = 3, label=r'VQMC method')
>>> dickerson = ax2.scatter(1.3989,-1.1645, s = 5, alpha=0.5, label = r'Experimental value')
...
>>> plt.xlabel(r'$s [r_b]$', fontsize=9)
>>> ylab = plt.ylabel(r'$\left\langle E \right\rangle [\mathrm{Hartree}]$', fontsize=9)
>>> legend = ax2.legend(loc='upper right', shadow=True, fontsize = 9)
>>> ax2.set_xlim([1, 2])
>>> ax2.set_ylim([-1.18, -1.05])
```

```python
>>> fig.savefig('s_closeup.pdf', bbox_inches='tight', pad_inches=0.1)
```

```python
>>> fig = plt.figure(figsize=(3, 1.85))
...
>>> s_row = np.append(np.linspace(1, 2, 11), 1.4011)
>>> s_row_finer = np.linspace(0.15, 0.95, 9)
>>> beta_en_1 = np.load('20160408_beta_graph_data.npy')
...
>>> ax2 = fig.add_subplot(111)
>>> s1_energy = ax2.plot(beta_en_1[:, 2], beta_en_1[:, 0], 'ok-', markersize = 3, label=r'VQMC method')
...
>>> plt.xlabel(r'$\beta$', fontsize=9)
>>> ylab = plt.ylabel(r'$\left\langle E \right\rangle [\mathrm{Hartree}]$', fontsize=9)
>>> ax2.set_ylim([-1.18, -1.08])
```

```python
>>> fig.savefig('beta_graph.pdf', bbox_inches='tight', pad_inches=0.1)
```
