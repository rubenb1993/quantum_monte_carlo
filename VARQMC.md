```python
>>> %%px
... import matplotlib.pyplot as plt
... import numpy as np
... from scipy.optimize import fsolve
... import ipyparallel as ipp
... %matplotlib inline
```

```python
>>> def Error(datavector, nblocks):
...
...     # Divide the datavector in nblocks and calculate the average value for each block
...     datavector1 = datavector[0:len(datavector) - len(datavector)%nblocks]
...     data_block = np.reshape(datavector1,(nblocks,-1))
...     # Used to data block specific heat
...     blockmean = np.mean(data_block,axis=1)
...     blockstd = np.std(data_block,axis=1)
...     # Calculate <A> en <A^2>
...     Mean = np.mean(blockmean)
...     # Standard deviation
...     std = np.std(blockmean)
...     return Mean, std, blockmean, blockstd
```

```python
>>> %%px
... def f(a):
...     """Coulomb cusp condition analytical expression"""
...     return (1/(1 + np.exp(-s/a)) - a)
```

```python
>>> %%px
... def normalize(vec):
...     absvec = np.linalg.norm(vec, axis=0)
...     return absvec, vec/absvec
```

```python
>>> %%px
... def apply_mask(mat, mask):
...     return mat[..., 1] * mask + mat[..., 0] * ~mask
```

```python
>>> %%px
... def simulate_harmonic_min(alpha,steps,x):
...     "A variational monte carlo method for a harmonic oscilator"
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
>>> def simulate_hydrogen_atom_min(alpha,steps,x):
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
>>> def simulate_helium_vector_min(alpha,steps,X,d):
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
>>> %%px
... def simulate_hydrogen_molecule_min(s,beta,steps,pos_walker):
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
... #         r_12_abs = apply_mask(r_12_abs, mask_r_abs).T
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
...                 dot_product + 1/s)
...         lnpsi[j,:] =  -r_12_abs**2/(2*(1+beta*r_12_abs)**2)
...     return Energy, lnpsi
```

## 1D harmonic oscilator

```python
>>> %%px
... numbalpha = 1
... alpha = 1.2
... beta = 0.6
... N = 400
... steps = 30000
... d = 0.05 #movement size
... diffmeanEn = 10
... meanEn = 0
... i = 0
...
... while diffmeanEn > 0.0001 and i < numbalpha:
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
... print("End result: alpha = ",alpha,", <E> = ", meanEn, 'var(E) = ', varE)
[stdout:0] 
alpha =  1.2 , <E> =  0.719980674955 var(E) =  0.47186710147
End result: alpha =  0.803473023975 , <E> =  0.719980674955 var(E) =  0.47186710147
[stdout:1] 
alpha =  1.2 , <E> =  0.706153220533 var(E) =  0.481500290188
End result: alpha =  0.795377907405 , <E> =  0.706153220533 var(E) =  0.481500290188
[stdout:2] 
alpha =  1.2 , <E> =  0.693129765077 var(E) =  0.515498701477
End result: alpha =  0.766807813885 , <E> =  0.693129765077 var(E) =  0.515498701477
[stdout:3] 
alpha =  1.2 , <E> =  0.694060568555 var(E) =  0.508687741842
End result: alpha =  0.772531309376 , <E> =  0.694060568555 var(E) =  0.508687741842
```

## Hydrogen Atom

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
>>> while diffmeanEn > 0.00001 and i < numbalpha:
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
```

## Helium Atom

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
```

## Hydrogen Molecule

```python
>>> c = ipp.Client()
>>> view = c[0:3]
>>> print(c.ids)
[0, 1, 2, 3]
```

```python
>>> %px simulate_harmonic_min(alpha,steps,x)
[0;31mOut[0:7]: [0m
(array([[ 0.73877844,  0.69487235,  0.6412346 , ...,  0.80347201,
          0.15581325,  0.27082155],
        [ 0.73932702,  0.69543685,  0.64499481, ...,  0.80326011,
          0.13213042,  0.25035922],
        [ 0.73901589,  0.70333847,  0.65052428, ...,  0.80225022,
          0.14999925,  0.28085396],
        ..., 
        [ 0.80320816, -0.83462605,  0.75108467, ...,  0.38626779,
          0.45693776,  0.80057651],
        [ 0.80305216, -0.8577351 ,  0.75041279, ...,  0.3805348 ,
          0.48181258,  0.80190857],
        [ 0.80346462, -0.80462695,  0.75473639, ...,  0.40142483,
          0.49732202,  0.8016401 ]]),
 array([[ -8.17741026e-02,  -1.37271504e-01,  -2.05069747e-01, ...,
          -1.27808854e-06,  -8.18643440e-01,  -6.73272685e-01],
        [ -8.10807005e-02,  -1.36557966e-01,  -2.00316827e-01, ...,
          -2.69126963e-04,  -8.48578600e-01,  -6.99137127e-01],
        [ -8.14739715e-02,  -1.26570304e-01,  -1.93327558e-01, ...,
          -1.54563236e-03,  -8.25992352e-01,  -6.60591702e-01],
        ..., 
        [ -3.34793495e-04,  -2.07056099e+00,  -6.62190031e-02, ...,
          -5.27348374e-01,  -4.38021375e-01,  -3.66119904e-03],
        [ -5.31974947e-04,  -2.09977089e+00,  -6.70682582e-02, ...,
          -5.34594891e-01,  -4.06579543e-01,  -1.97747300e-03],
        [ -1.06211961e-05,  -2.03264206e+00,  -6.16032147e-02, ...,
          -5.08189845e-01,  -3.86975576e-01,  -2.31681858e-03]]))
[0;31mOut[1:7]: [0m
(array([[ 0.75967526,  0.50487757,  0.06585795, ...,  0.66090781,
          0.76645465,  0.53384758],
        [ 0.76329911,  0.50998928,  0.06585795, ...,  0.66789591,
          0.77143673,  0.5276438 ],
        [ 0.75900088,  0.50989597,  0.06946445, ...,  0.6647612 ,
          0.76897657,  0.53122371],
        ..., 
        [ 0.77182538,  0.63570179,  0.79527001, ...,  0.79381743,
          0.58870183,  0.78019648],
        [ 0.76963179,  0.64601512,  0.79537639, ...,  0.7944285 ,
          0.60353255,  0.77449017],
        [ 0.76249839,  0.65323195,  0.79526888, ...,  0.7945667 ,
          0.59878731,  0.77692344]]),
 array([[ -4.66547611e-02,  -3.79613945e-01,  -9.53306788e-01, ...,
          -1.75720010e-01,  -3.77957304e-02,  -3.41757117e-01],
        [ -4.19192519e-02,  -3.72934164e-01,  -9.53306788e-01, ...,
          -1.66588248e-01,  -3.12853443e-02,  -3.49863956e-01],
        [ -4.75360141e-02,  -3.73056097e-01,  -9.48593967e-01, ...,
          -1.70684557e-01,  -3.45001910e-02,  -3.45185884e-01],
        ..., 
        [ -3.07774838e-02,  -2.08658208e-01,  -1.41000413e-04, ...,
          -2.03916879e-03,  -2.70075832e-01,  -1.98384685e-02],
        [ -3.36439728e-02,  -1.95181169e-01,  -1.98672694e-06, ...,
          -1.24064349e-03,  -2.50695657e-01,  -2.72952363e-02],
        [ -4.29656004e-02,  -1.85750511e-01,  -1.42469316e-04, ...,
          -1.06004641e-03,  -2.56896543e-01,  -2.41155387e-02]]))
[0;31mOut[2:7]: [0m
(array([[ 0.23142228,  0.23083913,  0.61092936, ...,  0.71816527,
          0.76566003,  0.31318386],
        [ 0.20095746,  0.25162544,  0.61981865, ...,  0.71252391,
          0.76600182,  0.30027967],
        [ 0.21658533,  0.24354189,  0.62348126, ...,  0.72051789,
          0.76565761,  0.27600929],
        ..., 
        [ 0.66852344,  0.67453291,  0.75520752, ...,  0.69251208,
          0.76276294,  0.633256  ],
        [ 0.66116454,  0.68040702,  0.750446  , ...,  0.69601832,
          0.75977666,  0.64172778],
        [ 0.65599949,  0.67244967,  0.75292777, ...,  0.69360673,
          0.7559421 ,  0.62729787]]),
 array([[-0.79200397, -0.79286663, -0.23059337, ..., -0.07195766,
         -0.00169793, -0.67105283],
        [-0.83707104, -0.76211713, -0.21744331, ..., -0.08030301,
         -0.00119231, -0.69014219],
        [-0.8139525 , -0.77407524, -0.21202515, ..., -0.06847739,
         -0.00170151, -0.72604573],
        ..., 
        [-0.14539357, -0.13650366, -0.01716049, ..., -0.1099068 ,
         -0.00598364, -0.19756524],
        [-0.1562797 , -0.12781401, -0.02420427, ..., -0.10471998,
         -0.01040129, -0.18503279],
        [-0.16392044, -0.13958544, -0.02053297, ..., -0.10828748,
         -0.01607381, -0.20637919]]))
[0;31mOut[3:7]: [0m
(array([[ 0.7679842 ,  0.66391873,  0.64642353, ...,  0.55176305,
          0.27743103,  0.77174053],
        [ 0.76618105,  0.67669465,  0.64293953, ...,  0.56577527,
          0.28545221,  0.77016811],
        [ 0.76449648,  0.68788513,  0.63557091, ...,  0.55624421,
          0.2719226 ,  0.77151725],
        ..., 
        [ 0.27233423,  0.77142208,  0.3190883 , ...,  0.24932006,
          0.4893585 ,  0.67629613],
        [ 0.2709845 ,  0.77087466,  0.30768123, ...,  0.27389166,
          0.49034851,  0.67249883],
        [ 0.29541519,  0.76957745,  0.33530144, ...,  0.27389166,
          0.50006321,  0.67259682]]),
 array([[-0.00655572, -0.15659044, -0.18181386, ..., -0.31828909,
         -0.71380288, -0.0011401 ],
        [-0.00915539, -0.13817097, -0.18683686, ..., -0.29808721,
         -0.70223847, -0.0034071 ],
        [-0.01158409, -0.12203726, -0.19746045, ..., -0.31182846,
         -0.72174458, -0.001462  ],
        ..., 
        [-0.72115111, -0.00159921, -0.65374419, ..., -0.75433142,
         -0.40825985, -0.13874553],
        [-0.72309707, -0.00238845, -0.67019014, ..., -0.71890571,
         -0.40683253, -0.14422022],
        [-0.6878745 , -0.00425868, -0.63036915, ..., -0.71890571,
         -0.39282651, -0.14407894]]))
```

```python
>>> %%px
... numbbeta = 1
... beta = 0.6
... zeta = 0.51
... N = 100
... steps = 300000
... d = 2.0
... s = 1.4
...
... meanEn = 0.0
... diffmeanEn = 10
... i = 0
...
... while diffmeanEn > 0.0001 and i < numbbeta:
...     pos_walker = np.random.uniform(-2,2,(N,3,2,2))
...     Energy = np.zeros(shape=(steps,N))
...     lnpsi = np.zeros(shape=(steps,N))
...
...     Energy, lnpsi = simulate_hydrogen_molecule_min(s, beta, steps, pos_walker)
...     meanEnNew = np.mean(Energy[7000:,:])
...     varE = np.var(Energy[7000:,:])
...
...     diffmeanEn = np.absolute(meanEnNew - meanEn)
...     meanEn = meanEnNew
...     print("beta = ",beta,", <E> = ", meanEn, "var(E) = ", varE)
...
...     meanlnpsi = np.mean(lnpsi[7000:,:])
...     meanEtimeslnpsi = np.mean(lnpsi[7000:,:]*Energy[7000:,:])
...     dEdbeta = 2*(meanEtimeslnpsi-meanEn*meanlnpsi)
...     #beta -= ((i+1)**(-zeta))*dEdbeta
...     beta -= 0.5*dEdbeta
...     i += 1
...
... print("End result: beta = ",beta,", <E> = ", meanEn, "var(E) = ", varE)
[stdout:0] 
beta =  0.6 , <E> =  -1.09584002886 var(E) =  0.0543803096636
End result: beta =  0.594689409624 , <E> =  -1.09584002886 var(E) =  0.0543803096636
[stdout:1] 
beta =  0.6 , <E> =  -1.09599174338 var(E) =  0.0544179740306
End result: beta =  0.594691420563 , <E> =  -1.09599174338 var(E) =  0.0544179740306
[stdout:2] 
beta =  0.6 , <E> =  -1.09605143529 var(E) =  0.0544038526925
End result: beta =  0.59464701062 , <E> =  -1.09605143529 var(E) =  0.0544038526925
[stdout:3] 
beta =  0.6 , <E> =  -1.09615406851 var(E) =  0.0543019261192
End result: beta =  0.594750124374 , <E> =  -1.09615406851 var(E) =  0.0543019261192
```

```python
>>> %%timeit
... numbbeta = 1
... beta = 0.6
... zeta = 0.51
... N = 100
... steps = 300000
... d = 2.0
... s = 1.4
...
... meanEn = 0.0
... diffmeanEn = 10
... i = 0
...
... while diffmeanEn > 0.0001 and i < numbbeta:
...     pos_walker = np.random.uniform(-2,2,(N,3,2,2))
...     Energy = np.zeros(shape=(steps,N))
...     lnpsi = np.zeros(shape=(steps,N))
...
...     Energy, lnpsi = simulate_hydrogen_molecule_min(s, beta, steps, pos_walker)
...     meanEnNew = np.mean(Energy[7000:,:])
...     varE = np.var(Energy[7000:,:])
...
...     diffmeanEn = np.absolute(meanEnNew - meanEn)
...     meanEn = meanEnNew
...     print("beta = ",beta,", <E> = ", meanEn, "var(E) = ", varE)
...
...     meanlnpsi = np.mean(lnpsi[7000:,:])
...     meanEtimeslnpsi = np.mean(lnpsi[7000:,:]*Energy[7000:,:])
...     dEdbeta = 2*(meanEtimeslnpsi-meanEn*meanlnpsi)
...     #beta -= ((i+1)**(-zeta))*dEdbeta
...     beta -= 0.5*dEdbeta
...     i += 1
...
... print("End result: beta = ",beta,", <E> = ", meanEn, "var(E) = ", varE)
[stdout:0] 
beta =  0.6 , <E> =  -1.09584002886 var(E) =  0.0543803096636
End result: beta =  0.594689409624 , <E> =  -1.09584002886 var(E) =  0.0543803096636
[stdout:1] 
beta =  0.6 , <E> =  -1.09599174338 var(E) =  0.0544179740306
End result: beta =  0.594691420563 , <E> =  -1.09599174338 var(E) =  0.0544179740306
[stdout:2] 
beta =  0.6 , <E> =  -1.09605143529 var(E) =  0.0544038526925
End result: beta =  0.59464701062 , <E> =  -1.09605143529 var(E) =  0.0544038526925
[stdout:3] 
beta =  0.6 , <E> =  -1.09615406851 var(E) =  0.0543019261192
End result: beta =  0.594750124374 , <E> =  -1.09615406851 var(E) =  0.0543019261192
```

```python
>>> plt.plot(Energy[10000:,1])
```
