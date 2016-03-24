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
```

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
...         dot_2 = r_12_hat/(2*a*(1 + beta*r_12_abs)**2)
...         dot_product = np.sum(dot_1*dot_2, axis=0)
...
...
...         #Energy = (-1/a**2 + (phi_1L/r_1L_abs + phi_1R/r_1R_abs)/(a*phi_1) + (phi_2L/r_2L_abs + phi_2R/r_2R_abs)/(a*phi_2) - \
...         #         1/r_1L_abs - 1/r_1R_abs - 1/r_2L_abs - 1/r_2R_abs + 1/r_12_abs - ((4*beta + 1)*r_12_abs + 4)/(4*r_12_abs*(1 + beta*r_12_abs)**4) + \
...         #        np.dot((phi_1L*r_1L_hat + phi_1R*r_1R_hat)/phi_1 - (phi_2L*r_2L_hat + phi_2R*r_2R_hat)/phi_2, r_12/(2*a*(1 + beta*r_12_abs)**2)))
...
...         Energy[j,:] = (-1/a**2 + (phi_1L/r_1L_abs + phi_1R/r_1R_abs)/(a*phi_1) + (phi_2L/r_2L_abs + phi_2R/r_2R_abs)/(a*phi_2) - \
...                  1/r_1L_abs - 1/r_1R_abs - 1/r_2L_abs - 1/r_2R_abs + 1/r_12_abs - ((4*beta + 1)*r_12_abs + 4)/(4*r_12_abs*(1 + beta*r_12_abs)**4) - \
...                 dot_product)
...     return Energy
```

---
scrolled: true
...

```python
>>> beta = [0.2,0.05]
>>> N = 400
>>> steps = 30000
>>> d = 0.1
>>> s = 1.4011
...
>>> meanEn = np.zeros(shape=(len(beta),))
>>> varE = np.zeros(shape=(len(beta),))
...
>>> for i in range(len(beta)):
...     pos_walker = np.random.uniform(-2,2,(N,3,2,2))
...     Energy = np.zeros(shape=(steps,N))
...     Energy = simulate_hydrogen_molecule(s, beta[i], steps, pos_walker)
...     meanEn[i] = np.mean(Energy[4000:,:])
...     varE[i] = np.var(Energy[4000:,:])
...
...     print("beta = ",beta[i],", <E> = ", meanEn[i], "var(E) = ", varE[i])
beta =  0.2 , <E> =  -2.48371885634 var(E) =  0.0875519866186
beta =  0.05 , <E> =  -2.94913117715 var(E) =  0.093558525628
```

```python
>>> plt.plot(Energy[7000:,1])
[<matplotlib.lines.Line2D at 0x12b0870b8>]
```

```python
>>> N = 400
>>> pos_walker = np.random.uniform(-2,2,(N,3,2,2))
>>> a = fsolve(f,0.1)
>>> beta = [0.1]
>>> steps = 30000
>>> d = 0.3
>>> s = 0.2
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
>>> r_12 = np.squeeze(-np.diff(pos_walker[...,0], axis=2)).T
>>> r_12_abs, r_12_hat = normalize(r_12)
>>> r_12_abs = r_12_abs
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
... dot_1 = -(phi_1L*r_1L_hat + phi_1R*r_1R_hat)/phi_1  + (phi_2L*r_2L_hat + phi_2R*r_2R_hat)/phi_2
>>> dot_2 = r_12_hat/(2*a*(1 + beta*r_12_abs)**2)
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
[[ 3.62944745  3.51586551]
 [ 3.1897865   3.2107211 ]
 [ 3.01788996  2.89917541]
 [ 1.70358551  1.60361376]
 [ 1.84733217  1.91000014]
 [ 2.13788159  2.18703151]
 [ 4.0772804   3.99131339]
 [ 2.80476097  2.71608146]
 [ 3.10402257  3.14687129]
 [ 1.78836594  1.66224996]
 [ 2.25111302  2.24295132]
 [ 2.79129279  2.7914808 ]
 [ 3.46581379  3.34582112]
 [ 1.85943744  1.75814524]
 [ 1.5648647   1.51025973]
 [ 1.59050316  1.4814626 ]
 [ 1.85005444  2.00915984]
 [ 1.79277574  1.88825382]
 [ 3.59417635  3.4634094 ]
 [ 2.21501585  2.29007324]
 [ 3.04030915  2.84842071]
 [ 1.51431451  1.6699591 ]
 [ 3.3923725   3.44407968]
 [ 3.06763286  3.0331713 ]
 [ 1.90528202  1.89524467]
 [ 2.91690974  2.7705131 ]
 [ 4.44486787  4.62773878]
 [ 3.10256371  3.17993406]
 [ 2.43167811  2.32340008]
 [ 2.25966198  2.37089071]
 [ 2.70164642  2.61352791]
 [ 2.25686629  2.09467548]
 [ 1.80575741  1.83540479]
 [ 3.22538974  3.27258397]
 [ 4.29593376  4.33965147]
 [ 1.97060345  2.16021243]
 [ 4.71815829  4.90679264]
 [ 3.00554237  3.06050485]
 [ 3.34063706  3.2226541 ]
 [ 3.38731907  3.66750984]
 [ 1.64580728  1.79335312]
 [ 4.36493008  4.3131588 ]
 [ 2.65999297  2.64643371]
 [ 4.32134046  4.21554852]
 [ 1.99705188  2.05290704]
 [ 2.81738647  2.93061467]
 [ 1.43208487  1.37589532]
 [ 3.15259007  3.2820862 ]
 [ 2.87533808  2.78944748]
 [ 3.92918906  3.87592264]
 [ 2.44472944  2.43293198]
 [ 2.61723182  2.54599772]
 [ 4.2867093   4.16327546]
 [ 3.20450238  3.25619096]
 [ 1.72804765  1.70748612]
 [ 2.80413041  3.00602584]
 [ 2.79603988  2.82258045]
 [ 2.75891677  2.84380573]
 [ 3.58238898  3.84791557]
 [ 3.70887129  3.71573802]
 [ 2.26326412  2.44200779]
 [ 4.23166755  4.02565731]
 [ 3.33417872  3.24556807]
 [ 2.03523946  2.02536704]
 [ 2.80682673  2.88786944]
 [ 2.2520411   2.21619097]
 [ 3.43131443  3.20589213]
 [ 3.61977898  3.44351003]
 [ 2.63813917  2.70855594]
 [ 3.12241776  3.12834921]
 [ 2.07486566  2.10379996]
 [ 3.65752992  3.598691  ]
 [ 2.51905335  2.66389755]
 [ 3.44863916  3.41289773]
 [ 1.6057084   1.58226578]
 [ 2.84149133  3.01845114]
 [ 2.97211078  2.91389621]
 [ 2.93745376  2.92519289]
 [ 3.01317288  3.1284845 ]
 [ 3.07046488  2.93573865]
 [ 1.93263486  1.88938615]
 [ 2.90874996  2.96225611]
 [ 1.26541234  1.28269933]
 [ 3.59492859  3.73917865]
 [ 4.2180721   4.0487309 ]
 [ 4.28811874  4.18689094]
 [ 2.86783068  3.03259948]
 [ 4.78136145  4.83178221]
 [ 1.84826216  1.87267793]
 [ 4.20985387  4.44553581]
 [ 5.29102611  5.32942705]
 [ 2.81370657  3.02231776]
 [ 2.17708978  2.13899017]
 [ 2.69194798  2.7422232 ]
 [ 1.54111313  1.56640835]
 [ 2.07227077  1.98124313]
 [ 2.61570108  2.62613033]
 [ 2.19620781  2.33527431]
 [ 2.15497839  2.15826364]
 [ 3.25399577  3.30113841]
 [ 2.66694365  2.72534985]
 [ 3.04065724  3.00233852]
 [ 3.6448586   3.5522687 ]
 [ 2.09942758  2.15452307]
 [ 2.7022494   2.80296656]
 [ 2.32368494  2.1863184 ]
 [ 3.81107911  4.09104559]
 [ 1.7320705   1.83677359]
 [ 2.15136734  2.39783403]
 [ 1.52520478  1.47960606]
 [ 3.55048519  3.64421548]
 [ 1.21152363  1.1348022 ]
 [ 4.92762279  4.73210999]
 [ 2.22819665  2.31503407]
 [ 2.11679551  2.05502052]
 [ 4.98642274  5.14467657]
 [ 3.17203077  3.07027106]
 [ 2.38346123  2.35385165]
 [ 3.54420692  3.62099364]
 [ 3.68828037  3.74882755]
 [ 3.33104632  3.32078243]
 [ 3.99238531  4.04712863]
 [ 3.45819663  3.6599488 ]
 [ 4.77961067  4.82948493]
 [ 2.00873174  2.14146953]
 [ 2.78856057  2.69025873]
 [ 1.84506738  1.67409324]
 [ 4.74437014  4.6687543 ]
 [ 3.39510066  3.28101988]
 [ 2.48866114  2.53040941]
 [ 4.89975335  4.9816218 ]
 [ 3.13939751  3.23923443]
 [ 2.84467805  2.77870586]
 [ 3.6027601   3.53391623]
 [ 1.32864665  1.25485833]
 [ 3.82157584  4.0941604 ]
 [ 3.19811588  3.10391929]
 [ 2.34069926  2.51143663]
 [ 3.59477513  3.64825532]
 [ 3.59212808  3.43548982]
 [ 2.78690002  2.8299979 ]
 [ 3.27657181  3.1417747 ]
 [ 2.04130377  2.08010267]
 [ 5.34554215  5.16754929]
 [ 2.49541065  2.48865059]
 [ 4.15008142  4.20623334]
 [ 3.18913264  3.18018412]
 [ 3.24712264  3.25413078]
 [ 2.62994177  2.61630563]
 [ 3.00507553  2.97588263]
 [ 4.27975011  4.21731097]
 [ 3.80821989  3.69519547]
 [ 3.85281407  3.78409644]
 [ 3.56803355  3.56462658]
 [ 2.97070077  2.91830916]
 [ 2.14057778  2.05914341]
 [ 2.34077463  2.47448517]
 [ 3.48761674  3.44068008]
 [ 1.75964342  1.77341532]
 [ 1.65078276  1.65935215]
 [ 2.94055977  3.00189696]
 [ 2.82914191  2.90630113]
 [ 2.47114508  2.55616379]
 [ 3.94026013  4.00168538]
 [ 2.50527947  2.3247128 ]
 [ 4.73481872  4.74939139]
 [ 2.1673082   2.24689653]
 [ 2.83768001  2.8415162 ]
 [ 1.80197366  1.79736829]
 [ 3.13535269  3.20628365]
 [ 2.18153718  2.3448299 ]
 [ 2.97504014  2.93414947]
 [ 2.09916667  2.10381408]
 [ 3.80875632  3.88388892]
 [ 2.97214223  2.89124707]
 [ 2.19381531  2.26048125]
 [ 3.15190309  3.12676481]
 [ 3.05727054  3.06906932]
 [ 3.29385905  3.14467405]
 [ 3.9386053   4.0187733 ]
 [ 2.7077204   2.79137467]
 [ 1.62273079  1.77920114]
 [ 2.89191607  2.84845788]
 [ 2.74611639  2.77659709]
 [ 1.94098221  1.83657009]
 [ 4.78114063  4.70285856]
 [ 2.31811406  2.35718935]
 [ 2.16615176  2.24077793]
 [ 2.88472871  2.93755642]
 [ 3.66718302  3.74777135]
 [ 3.84568631  4.19928183]
 [ 2.63445802  2.75006101]
 [ 2.85564435  2.92323321]
 [ 2.4174243   2.43764413]
 [ 4.51298254  4.61132944]
 [ 3.05224908  3.10715595]
 [ 3.44839236  3.51850946]
 [ 1.89460321  1.76732494]
 [ 2.71647497  2.67082631]
 [ 2.62731171  2.59549267]
 [ 3.59801492  3.83790739]
 [ 2.81433309  2.82331789]
 [ 2.88051223  2.86240272]
 [ 4.01020234  3.9992893 ]
 [ 3.36252053  3.53004961]
 [ 2.69294838  2.60476513]
 [ 2.71470841  2.6819281 ]
 [ 2.39403126  2.39608571]
 [ 2.68101524  2.64995298]
 [ 3.69868076  3.58606874]
 [ 4.38192782  4.32104924]
 [ 2.90866595  2.93089409]
 [ 1.26421295  1.41905605]
 [ 3.58850408  3.49680413]
 [ 2.96495139  3.02229213]
 [ 2.31331863  2.21998528]
 [ 3.20004387  3.1811154 ]
 [ 2.55578582  2.73495464]
 [ 3.82050651  3.69034446]
 [ 3.7749051   3.84083766]
 [ 2.96108543  2.99821763]
 [ 2.27047168  2.36710257]
 [ 2.20900303  2.35092393]
 [ 4.3749004   4.34001473]
 [ 3.51244392  3.55803724]
 [ 2.14714727  1.99133311]
 [ 1.16670774  1.30294868]
 [ 2.68994627  2.67738448]
 [ 3.63059902  3.62724033]
 [ 2.27069416  2.18202632]
 [ 3.50545812  3.44214164]
 [ 1.88528014  1.92391322]
 [ 2.57707697  2.52719087]
 [ 4.35998914  4.44311919]
 [ 3.29385887  3.42909872]
 [ 2.39822822  2.51997684]
 [ 3.23371301  3.14626189]
 [ 3.48798888  3.50082221]
 [ 2.56929302  2.52612255]
 [ 3.04901429  3.01017689]
 [ 2.34025143  2.46022405]
 [ 3.35344632  3.42086677]
 [ 2.93439396  2.76078923]
 [ 3.12113949  3.19139899]
 [ 3.54308287  3.84397343]
 [ 2.04131708  2.00664194]
 [ 2.63269393  2.76779215]
 [ 4.18083966  4.01880445]
 [ 3.58062337  3.66416527]
 [ 2.95893734  2.86590959]
 [ 3.21911759  3.42171251]
 [ 2.61702063  2.5477458 ]
 [ 2.61342252  2.64797557]
 [ 2.54267953  2.66326597]
 [ 2.21649394  2.30672782]
 [ 2.57788622  2.65064603]
 [ 2.40322971  2.4587816 ]
 [ 3.8230168   3.93867434]
 [ 2.44324938  2.40071755]
 [ 2.23714724  2.26830652]
 [ 2.64359126  2.91519129]
 [ 4.3687108   4.38019457]
 [ 2.96280147  2.94837792]
 [ 2.12395909  2.20977109]
 [ 3.11793171  2.87818133]
 [ 4.27067739  4.17956378]
 [ 2.26979153  2.39529514]
 [ 3.36907444  3.47763562]
 [ 2.0773854   2.06447986]
 [ 2.2630632   2.44170687]
 [ 2.48955987  2.6524626 ]
 [ 3.87899609  3.85400383]
 [ 2.73595229  2.75424904]
 [ 2.57065065  2.54813761]
 [ 1.850127    1.92094528]
 [ 2.67570994  2.68127932]
 [ 2.22532541  2.19051616]
 [ 4.01107102  4.19178938]
 [ 1.48170553  1.53870932]
 [ 2.61816908  2.60101486]
 [ 3.31714875  3.45795461]
 [ 1.63693385  1.68730364]
 [ 3.70454846  3.93547762]
 [ 2.97039053  2.9826017 ]
 [ 3.90036331  4.02184668]
 [ 4.17346344  4.01700711]
 [ 2.7110665   2.66067158]
 [ 4.03823932  4.03190363]
 [ 3.80893065  3.86471206]
 [ 3.16358469  3.1988623 ]
 [ 4.39986361  4.30846612]
 [ 3.70227142  3.60452891]
 [ 3.56733326  3.64869866]
 [ 1.61305537  1.61118149]
 [ 2.67639041  2.54476127]
 [ 1.23903041  1.19775133]
 [ 3.26221688  3.26598058]
 [ 2.7205792   2.45739578]
 [ 2.93926212  2.95740914]
 [ 3.5364618   3.66964981]
 [ 2.47774182  2.49892086]
 [ 2.21368644  2.30776992]
 [ 3.16750929  3.33720075]
 [ 4.05417038  4.07944759]
 [ 3.02223271  2.95517496]
 [ 2.84633022  2.92819864]
 [ 2.91544813  2.99298871]
 [ 1.43737787  1.54571363]
 [ 4.20687725  4.21654386]
 [ 3.421465    3.42320983]
 [ 2.90244361  3.06376522]
 [ 3.62123896  3.59808362]
 [ 2.63707099  2.74450122]
 [ 3.38908737  3.11599667]
 [ 3.1151875   2.97451453]
 [ 2.25312121  2.29495182]
 [ 4.47601278  4.29211513]
 [ 2.94878774  3.1063504 ]
 [ 4.22740306  4.30723653]
 [ 3.33886938  3.24763084]
 [ 2.13712401  2.2038694 ]
 [ 3.8813118   3.87706973]
 [ 1.6617313   1.73550111]
 [ 4.17257679  4.28698455]
 [ 4.40879575  4.33237154]
 [ 3.11671238  3.1158927 ]
 [ 3.54069894  3.57791654]
 [ 4.0475427   4.02062845]
 [ 1.39223462  1.47887815]
 [ 3.25654222  3.40785183]
 [ 3.52349328  3.48421028]
 [ 3.34879305  3.15571964]
 [ 2.02221506  2.01646786]
 [ 2.08542725  2.0576507 ]
 [ 1.80018689  1.80889595]
 [ 2.6944775   2.86273796]
 [ 3.54091668  3.68882483]
 [ 3.68594612  3.48647742]
 [ 1.96985304  1.92670021]
 [ 1.97816045  1.79678558]
 [ 3.20549641  3.10891636]
 [ 4.42070682  4.45444744]
 [ 1.73557648  1.69442174]
 [ 1.68805537  1.70338221]
 [ 2.58583701  2.64232856]
 [ 3.63708578  3.61176521]
 [ 3.56082726  3.4895789 ]
 [ 3.26232984  3.23364564]
 [ 2.17941151  2.1734834 ]
 [ 2.29137854  2.43016994]
 [ 2.7336455   2.76107966]
 [ 1.2921156   1.34472604]
 [ 2.93906318  2.8146317 ]
 [ 3.41118681  3.37253709]
 [ 2.13751905  2.0653649 ]
 [ 2.89661798  3.07870795]
 [ 2.95026235  3.00233841]
 [ 3.28528031  3.24821523]
 [ 3.57547267  3.22330831]
 [ 3.87302534  3.75775501]
 [ 3.22516407  3.0862709 ]
 [ 2.65208848  2.43083291]
 [ 1.95933924  1.93616013]
 [ 2.99418769  3.18620784]
 [ 2.95447085  3.0190651 ]
 [ 3.57132229  3.49450309]
 [ 3.30848648  3.19445652]
 [ 2.61163908  2.52804759]
 [ 4.7118288   4.62511135]
 [ 3.6889111   3.77710752]
 [ 2.30664439  2.33069265]
 [ 4.56686234  4.4995305 ]
 [ 2.31328149  2.34047033]
 [ 2.40584573  2.23883137]
 [ 2.32617118  2.14908463]
 [ 1.99432882  1.90466325]
 [ 2.57557239  2.61534288]
 [ 2.75476353  2.87711679]
 [ 2.08749611  2.13503933]
 [ 3.61591542  3.47121049]
 [ 2.90183228  2.68582793]
 [ 3.01697806  2.92713246]
 [ 2.40698322  2.31149083]
 [ 3.38631811  3.61500286]
 [ 2.44498097  2.49597617]
 [ 1.91718995  1.91133706]
 [ 3.22704488  3.02915951]
 [ 1.69287376  1.59524609]
 [ 3.44955828  3.39842108]
 [ 2.49963253  2.52317186]
 [ 3.75886647  3.89808391]
 [ 3.78228144  3.80133558]
 [ 2.10239415  1.86126968]
 [ 2.71560986  2.61947947]
 [ 2.99236321  2.89803704]
 [ 2.87559335  3.0250533 ]
 [ 2.5372436   2.51175715]
 [ 2.78101335  2.6392656 ]
 [ 2.5923791   2.64122451]
 [ 3.36660069  3.45950172]]
(400,)
```

```python
>>> beta=1
>>> d = 0.1
>>> pos_walker = np.zeros(shape=(2,3,2,2))
>>> pos_walker[...,0] = np.transpose(np.array([[[0,0,0],[1,0,0]],[[1,0,0],[0,0,0]]]), axes = [1,2,0])
>>> pos_walker[...,1] = pos_walker[...,0] + (np.random.rand(2,3,2) - 0.5)*d
>>> offset_array = np.append(1.4/2*np.ones([2,1,2,2]),np.zeros([2,2,2,2]), axis=1)
>>> left_right_array = np.append(pos_walker + offset_array, pos_walker - offset_array, axis=2)
>>> phi_1L, phi_2L, phi_1R, phi_2R = np.transpose(np.exp(np.linalg.norm(left_right_array,axis=1)/-a), axes=[1, 0, 2])
...
>>> phi_1 = phi_1L + phi_1R
>>> phi_2 = phi_2L + phi_2R
...
>>> r_12 = -np.diff(pos_walker, axis=2)
>>> r_12_abs = np.linalg.norm(r_12, axis=1)
...
>>> psi_jastrow = np.squeeze(np.exp(r_12_abs/(2*(1+beta*r_12_abs))))
>>> r_12_abs = np.linalg.norm(r_12, axis=1)
...
>>> psi = phi_1*phi_2*psi_jastrow
...
>>> p = (psi[:,1]/psi[:,0]) ** 2
>>> mask = p > np.random.rand(2)
>>> mask = np.array([False, True]) #test mask
>>> mask_walker = np.tile(mask,(2,3,1)).T
>>> mask_left_right = np.tile(mask,(4,3,1)).T
>>> mask_r_abs = np.tile(mask,(1,1)).T
>>> mask_r_12 = np.tile(mask,(1,3,1)).T
>>> r_1L, r_2L, r_1R, r_2R = apply_mask(left_right_array, mask_left_right).T
...
>>> phi_1L = apply_mask(phi_1L, mask).T
>>> phi_2L = apply_mask(phi_2L, mask).T
>>> phi_1R = apply_mask(phi_1R, mask).T
>>> phi_2R = apply_mask(phi_2R, mask).T
>>> phi_1 = phi_1L + phi_1R
>>> phi_2 = phi_2L + phi_2R
...
>>> r_12 = np.squeeze(-np.diff(pos_walker[...,0], axis=2)).T
>>> r_12_abs, r_12_hat = normalize(r_12)
>>> r_12_abs = r_12_abs
>>> r_12_hat = r_12_hat
...
>>> pos_walker[...,0] = apply_mask(pos_walker, mask_walker)
...
>>> r_1L_abs, r_1L_hat = normalize(r_1L)
>>> r_2L_abs, r_2L_hat = normalize(r_2L)
>>> r_1R_abs, r_1R_hat = normalize(r_1R)
>>> r_2R_abs, r_2R_hat = normalize(r_2R)
...
...
>>> dot_1 = -((phi_1L*r_1L_hat + phi_1R*r_1R_hat)/phi_1 - (phi_2L*r_2L_hat + phi_2R*r_2R_hat)/phi_2)
>>> dot_2 = r_12_hat/(2*a*(1 + beta*r_12_abs*r_12_abs))
>>> dot_product = np.sum(dot_1*dot_2, axis=0)
[[ 1.28402542  1.28487336]
 [ 1.28402542  1.28807361]]
```

```python

```
