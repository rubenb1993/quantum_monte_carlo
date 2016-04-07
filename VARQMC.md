```python
>>> import matplotlib.pyplot as plt
>>> import numpy as np
>>> from scipy.optimize import fsolve
...
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
End result: alpha =  0.999987398533 iteration # =  72
(26000, 400)
<E> =  -0.500000009499 with error:  5.79611671979e-08
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
alpha =  0.5 , <E> =  -2.88440663477 , dalpha(%) =  100
alpha =  0.497500714947 , <E> =  -2.87611969139 , dalpha(%) =  -0.499857010672
alpha =  0.495037990861 , <E> =  -2.87804591076 , dalpha(%) =  -0.495019205384
alpha =  0.492520990669 , <E> =  -2.8782143135 , dalpha(%) =  -0.508445864493
alpha =  0.490030478986 , <E> =  -2.8755775907 , dalpha(%) =  -0.505666099473
alpha =  0.487537843025 , <E> =  -2.87935566075 , dalpha(%) =  -0.508669576208
alpha =  0.484998921297 , <E> =  -2.87683725383 , dalpha(%) =  -0.520764031836
alpha =  0.482508033443 , <E> =  -2.88570360883 , dalpha(%) =  -0.513586266876
alpha =  0.479981899853 , <E> =  -2.88512773873 , dalpha(%) =  -0.523542286276
alpha =  0.477470493506 , <E> =  -2.88043836434 , dalpha(%) =  -0.523229385898
alpha =  0.474953697551 , <E> =  -2.87638519789 , dalpha(%) =  -0.527110258918
alpha =  0.472451590094 , <E> =  -2.88242325574 , dalpha(%) =  -0.526810817707
alpha =  0.469929439843 , <E> =  -2.88230636734 , dalpha(%) =  -0.53384310761
alpha =  0.46736025117 , <E> =  -2.88225231194 , dalpha(%) =  -0.546717965551
alpha =  0.464791837314 , <E> =  -2.87988057434 , dalpha(%) =  -0.549557616283
alpha =  0.462288262488 , <E> =  -2.87994150707 , dalpha(%) =  -0.538644318906
alpha =  0.459750450946 , <E> =  -2.88403472462 , dalpha(%) =  -0.548967332188
alpha =  0.457210415313 , <E> =  -2.88000189676 , dalpha(%) =  -0.552481379452
alpha =  0.454638477347 , <E> =  -2.88801472458 , dalpha(%) =  -0.562528297727
alpha =  0.452055060791 , <E> =  -2.88933816956 , dalpha(%) =  -0.568235352678
alpha =  0.44947088411 , <E> =  -2.88216705434 , dalpha(%) =  -0.57165086853
alpha =  0.44692355552 , <E> =  -2.88261638649 , dalpha(%) =  -0.566739399633
alpha =  0.444354130828 , <E> =  -2.88377672439 , dalpha(%) =  -0.574913687235
alpha =  0.441764145921 , <E> =  -2.88197139672 , dalpha(%) =  -0.582865045471
alpha =  0.439201026235 , <E> =  -2.88384936243 , dalpha(%) =  -0.580200930807
alpha =  0.436631931954 , <E> =  -2.88270486211 , dalpha(%) =  -0.584947239994
alpha =  0.434025728903 , <E> =  -2.88374816894 , dalpha(%) =  -0.596887872856
alpha =  0.431432415916 , <E> =  -2.88364179045 , dalpha(%) =  -0.597502132697
alpha =  0.428874826371 , <E> =  -2.88701197318 , dalpha(%) =  -0.592813486096
alpha =  0.426241125576 , <E> =  -2.88291713878 , dalpha(%) =  -0.614095449959
alpha =  0.42364956115 , <E> =  -2.88476218834 , dalpha(%) =  -0.608004312724
alpha =  0.421059271236 , <E> =  -2.88008265735 , dalpha(%) =  -0.611422777541
alpha =  0.418504569463 , <E> =  -2.88214356414 , dalpha(%) =  -0.606732103427
alpha =  0.415870293762 , <E> =  -2.88265247189 , dalpha(%) =  -0.62944968676
alpha =  0.413277121671 , <E> =  -2.88619738582 , dalpha(%) =  -0.623553095599
alpha =  0.410666808689 , <E> =  -2.88362793995 , dalpha(%) =  -0.631613231245
alpha =  0.408033048959 , <E> =  -2.88716972333 , dalpha(%) =  -0.641337374802
alpha =  0.40543922909 , <E> =  -2.88706096526 , dalpha(%) =  -0.635688671797
alpha =  0.402866152699 , <E> =  -2.88044530527 , dalpha(%) =  -0.634639227254
alpha =  0.400217647326 , <E> =  -2.8859451219 , dalpha(%) =  -0.657415708778
alpha =  0.397610444235 , <E> =  -2.88941139601 , dalpha(%) =  -0.651446308868
alpha =  0.39498425005 , <E> =  -2.89238712227 , dalpha(%) =  -0.660494265895
alpha =  0.392302061096 , <E> =  -2.88362502048 , dalpha(%) =  -0.679062254817
alpha =  0.389702759047 , <E> =  -2.88483981941 , dalpha(%) =  -0.662576699545
alpha =  0.38708132813 , <E> =  -2.88574045776 , dalpha(%) =  -0.672674456665
alpha =  0.384418112989 , <E> =  -2.88981354783 , dalpha(%) =  -0.688024698501
alpha =  0.381716501686 , <E> =  -2.88414667297 , dalpha(%) =  -0.702779398869
alpha =  0.379073084 , <E> =  -2.88794311276 , dalpha(%) =  -0.692508098137
alpha =  0.376528436034 , <E> =  -2.8854433218 , dalpha(%) =  -0.671281626852
alpha =  0.37391627942 , <E> =  -2.88771301585 , dalpha(%) =  -0.693747500711
alpha =  0.371248473144 , <E> =  -2.88600764316 , dalpha(%) =  -0.713476899262
alpha =  0.36861709777 , <E> =  -2.88555066207 , dalpha(%) =  -0.708790894397
alpha =  0.365996659141 , <E> =  -2.88975085667 , dalpha(%) =  -0.710883636506
alpha =  0.36339893884 , <E> =  -2.88877629255 , dalpha(%) =  -0.709766123745
alpha =  0.360699898308 , <E> =  -2.891257589 , dalpha(%) =  -0.742721082593
alpha =  0.358106371164 , <E> =  -2.88541260361 , dalpha(%) =  -0.719026303141
alpha =  0.35557779665 , <E> =  -2.88873500012 , dalpha(%) =  -0.706095930574
alpha =  0.352943415678 , <E> =  -2.89302895698 , dalpha(%) =  -0.740873304279
alpha =  0.35027177341 , <E> =  -2.88656425206 , dalpha(%) =  -0.756960506828
alpha =  0.347707552658 , <E> =  -2.88661360117 , dalpha(%) =  -0.732066054608
alpha =  0.345156862467 , <E> =  -2.88507640829 , dalpha(%) =  -0.733573421686
alpha =  0.342571048844 , <E> =  -2.88337448966 , dalpha(%) =  -0.749170566842
alpha =  0.339957540824 , <E> =  -2.88787773342 , dalpha(%) =  -0.76290977556
alpha =  0.337338495179 , <E> =  -2.88562988026 , dalpha(%) =  -0.770403750596
alpha =  0.334805372344 , <E> =  -2.88786781565 , dalpha(%) =  -0.750914251179
alpha =  0.332233526056 , <E> =  -2.89079067235 , dalpha(%) =  -0.768161594902
alpha =  0.329685211621 , <E> =  -2.88777251588 , dalpha(%) =  -0.767025069805
alpha =  0.327106967306 , <E> =  -2.8848992462 , dalpha(%) =  -0.782032139616
alpha =  0.324554924949 , <E> =  -2.89154891651 , dalpha(%) =  -0.780185875698
alpha =  0.321923061747 , <E> =  -2.88748780651 , dalpha(%) =  -0.810914578643
alpha =  0.319410733834 , <E> =  -2.89219229398 , dalpha(%) =  -0.780412530719
alpha =  0.316855122979 , <E> =  -2.88645937574 , dalpha(%) =  -0.800101744707
alpha =  0.314269572295 , <E> =  -2.89037087432 , dalpha(%) =  -0.816004065083
alpha =  0.311716337489 , <E> =  -2.89061985854 , dalpha(%) =  -0.812434620361
alpha =  0.30917866823 , <E> =  -2.88925876806 , dalpha(%) =  -0.814095686786
alpha =  0.306633709082 , <E> =  -2.88869888771 , dalpha(%) =  -0.82313542625
alpha =  0.304142860673 , <E> =  -2.88831735784 , dalpha(%) =  -0.812320477241
alpha =  0.301657955184 , <E> =  -2.89420052182 , dalpha(%) =  -0.817019174317
alpha =  0.299150223805 , <E> =  -2.89078079426 , dalpha(%) =  -0.831316176507
alpha =  0.29666411948 , <E> =  -2.89476932651 , dalpha(%) =  -0.831055478747
alpha =  0.294184465478 , <E> =  -2.88977054515 , dalpha(%) =  -0.835845604172
alpha =  0.291667189976 , <E> =  -2.89132884691 , dalpha(%) =  -0.855679275152
alpha =  0.289244187446 , <E> =  -2.88983855693 , dalpha(%) =  -0.83074223432
alpha =  0.286692542026 , <E> =  -2.88794146057 , dalpha(%) =  -0.882176904509
alpha =  0.284225799879 , <E> =  -2.89063264808 , dalpha(%) =  -0.860413783141
alpha =  0.281753978358 , <E> =  -2.89101967129 , dalpha(%) =  -0.869668243412
alpha =  0.279377996937 , <E> =  -2.89103801787 , dalpha(%) =  -0.843282297428
alpha =  0.27695622396 , <E> =  -2.88913452473 , dalpha(%) =  -0.866844562907
alpha =  0.274564211272 , <E> =  -2.88700966731 , dalpha(%) =  -0.863678979226
alpha =  0.272218898383 , <E> =  -2.89079765792 , dalpha(%) =  -0.854194681001
alpha =  0.269877781075 , <E> =  -2.8938211443 , dalpha(%) =  -0.860012777224
alpha =  0.267546393944 , <E> =  -2.89017724548 , dalpha(%) =  -0.863867756024
alpha =  0.265213671754 , <E> =  -2.88906614054 , dalpha(%) =  -0.871894461167
alpha =  0.262908790726 , <E> =  -2.89064017253 , dalpha(%) =  -0.8690656905
alpha =  0.260504809232 , <E> =  -2.89179725092 , dalpha(%) =  -0.914378514322
alpha =  0.258225748126 , <E> =  -2.89055355032 , dalpha(%) =  -0.874863351884
alpha =  0.255937599178 , <E> =  -2.88862886607 , dalpha(%) =  -0.886104102644
alpha =  0.253624554448 , <E> =  -2.89078930524 , dalpha(%) =  -0.903753390255
alpha =  0.251275125208 , <E> =  -2.8924220752 , dalpha(%) =  -0.926341396656
alpha =  0.248893474109 , <E> =  -2.88713885917 , dalpha(%) =  -0.947826052111
alpha =  0.246709404393 , <E> =  -2.89281189814 , dalpha(%) =  -0.877511844636
alpha =  0.244500963256 , <E> =  -2.88852449538 , dalpha(%) =  -0.895158878365
alpha =  0.242122569293 , <E> =  -2.88688642618 , dalpha(%) =  -0.972754434719
alpha =  0.239929900709 , <E> =  -2.89480568104 , dalpha(%) =  -0.905602724528
alpha =  0.237683081744 , <E> =  -2.88945355412 , dalpha(%) =  -0.936448086936
alpha =  0.235512358534 , <E> =  -2.88828565123 , dalpha(%) =  -0.913284695685
alpha =  0.2334400611 , <E> =  -2.89120095979 , dalpha(%) =  -0.879910271704
alpha =  0.23129998481 , <E> =  -2.8944905043 , dalpha(%) =  -0.916756224075
alpha =  0.229086112117 , <E> =  -2.89218049468 , dalpha(%) =  -0.957143466853
alpha =  0.226981553175 , <E> =  -2.89245354717 , dalpha(%) =  -0.918675917105
alpha =  0.224859096126 , <E> =  -2.89514580306 , dalpha(%) =  -0.935079093121
alpha =  0.222702471096 , <E> =  -2.8922937027 , dalpha(%) =  -0.959100640127
alpha =  0.220710222807 , <E> =  -2.89105874796 , dalpha(%) =  -0.894578438986
alpha =  0.218727863247 , <E> =  -2.8924391289 , dalpha(%) =  -0.898172968194
alpha =  0.216647319084 , <E> =  -2.88977120866 , dalpha(%) =  -0.951202161758
alpha =  0.214790674145 , <E> =  -2.89129416411 , dalpha(%) =  -0.856989574872
alpha =  0.212856344015 , <E> =  -2.89299457693 , dalpha(%) =  -0.900565230567
alpha =  0.210946913014 , <E> =  -2.89138518483 , dalpha(%) =  -0.897051487822
alpha =  0.209138605574 , <E> =  -2.89180005734 , dalpha(%) =  -0.857233421376
alpha =  0.207197735236 , <E> =  -2.88805294425 , dalpha(%) =  -0.928030639407
alpha =  0.205312226293 , <E> =  -2.8920201276 , dalpha(%) =  -0.910004610115
alpha =  0.203419101517 , <E> =  -2.89160240881 , dalpha(%) =  -0.922071135563
alpha =  0.201574567738 , <E> =  -2.89159128249 , dalpha(%) =  -0.906765276812
alpha =  0.199872453035 , <E> =  -2.89247332003 , dalpha(%) =  -0.844409451841
alpha =  0.198117299658 , <E> =  -2.89143611798 , dalpha(%) =  -0.878136706834
alpha =  0.196402868124 , <E> =  -2.89550544488 , dalpha(%) =  -0.865361852605
alpha =  0.194584916281 , <E> =  -2.8920113843 , dalpha(%) =  -0.92562387736
alpha =  0.19291587405 , <E> =  -2.88941096161 , dalpha(%) =  -0.857744918169
alpha =  0.191395696317 , <E> =  -2.88930486592 , dalpha(%) =  -0.788000334423
alpha =  0.189834709657 , <E> =  -2.89316501942 , dalpha(%) =  -0.815580856734
alpha =  0.188258562177 , <E> =  -2.89055010791 , dalpha(%) =  -0.830273601374
alpha =  0.186644024332 , <E> =  -2.89105374589 , dalpha(%) =  -0.857617218399
alpha =  0.185186232909 , <E> =  -2.89049001587 , dalpha(%) =  -0.781054431498
alpha =  0.183628883747 , <E> =  -2.88972042649 , dalpha(%) =  -0.840963789751
alpha =  0.182089905622 , <E> =  -2.89550285978 , dalpha(%) =  -0.838091531634
alpha =  0.180682853666 , <E> =  -2.89403377484 , dalpha(%) =  -0.77272375538
alpha =  0.17924521662 , <E> =  -2.88925839922 , dalpha(%) =  -0.795668773326
alpha =  0.17788534916 , <E> =  -2.89060500947 , dalpha(%) =  -0.758663179765
alpha =  0.176538482763 , <E> =  -2.89144099819 , dalpha(%) =  -0.757154202898
alpha =  0.175144617139 , <E> =  -2.89477937534 , dalpha(%) =  -0.789553417162
alpha =  0.173758773299 , <E> =  -2.89401035098 , dalpha(%) =  -0.791256884337
alpha =  0.172370266123 , <E> =  -2.89047541064 , dalpha(%) =  -0.799100470779
alpha =  0.171071294442 , <E> =  -2.89071698787 , dalpha(%) =  -0.753593824969
alpha =  0.169871726493 , <E> =  -2.88856620171 , dalpha(%) =  -0.701209371714
alpha =  0.168666593315 , <E> =  -2.88722119338 , dalpha(%) =  -0.709437175398
alpha =  0.167524912152 , <E> =  -2.88967137142 , dalpha(%) =  -0.676886358918
alpha =  0.166405676047 , <E> =  -2.89254865221 , dalpha(%) =  -0.66810129398
alpha =  0.165093582878 , <E> =  -2.88954090968 , dalpha(%) =  -0.788490632944
alpha =  0.163955178723 , <E> =  -2.89135378359 , dalpha(%) =  -0.689550820278
alpha =  0.162865740503 , <E> =  -2.88966766486 , dalpha(%) =  -0.664473198623
alpha =  0.16178355277 , <E> =  -2.88910799937 , dalpha(%) =  -0.664466160763
alpha =  0.160669121606 , <E> =  -2.88840566123 , dalpha(%) =  -0.688840827756
alpha =  0.159674547035 , <E> =  -2.88589332014 , dalpha(%) =  -0.619020357544
alpha =  0.158864800761 , <E> =  -2.88771247316 , dalpha(%) =  -0.507122950206
alpha =  0.157903692897 , <E> =  -2.88965924333 , dalpha(%) =  -0.604984779374
alpha =  0.156985507707 , <E> =  -2.88864903995 , dalpha(%) =  -0.581484303741
alpha =  0.156076510316 , <E> =  -2.89569850907 , dalpha(%) =  -0.579032679111
alpha =  0.155182506295 , <E> =  -2.88841549289 , dalpha(%) =  -0.572798571058
alpha =  0.15437656407 , <E> =  -2.89447720924 , dalpha(%) =  -0.519351211069
alpha =  0.153404594147 , <E> =  -2.89368587243 , dalpha(%) =  -0.629609765496
alpha =  0.152482246707 , <E> =  -2.89236876062 , dalpha(%) =  -0.60125151033
alpha =  0.151691125321 , <E> =  -2.89163586058 , dalpha(%) =  -0.518828521855
alpha =  0.150927460467 , <E> =  -2.8875853954 , dalpha(%) =  -0.503434101257
alpha =  0.150189092621 , <E> =  -2.88974141114 , dalpha(%) =  -0.489220347528
alpha =  0.149448317186 , <E> =  -2.89305465108 , dalpha(%) =  -0.493228517281
alpha =  0.148752048021 , <E> =  -2.88879010896 , dalpha(%) =  -0.465892943965
alpha =  0.148193435057 , <E> =  -2.8912614739 , dalpha(%) =  -0.375532956963
alpha =  0.147573831203 , <E> =  -2.88656729756 , dalpha(%) =  -0.418104792553
alpha =  0.146943722841 , <E> =  -2.89114221411 , dalpha(%) =  -0.426978385633
alpha =  0.146278107325 , <E> =  -2.88710168338 , dalpha(%) =  -0.452973085707
alpha =  0.145661968644 , <E> =  -2.89240793655 , dalpha(%) =  -0.421210454846
alpha =  0.144963285295 , <E> =  -2.8877802669 , dalpha(%) =  -0.479660789619
alpha =  0.14433573861 , <E> =  -2.88966454193 , dalpha(%) =  -0.432900429283
alpha =  0.143845878882 , <E> =  -2.88913483505 , dalpha(%) =  -0.33938907536
alpha =  0.143266895187 , <E> =  -2.88804573251 , dalpha(%) =  -0.402502803421
alpha =  0.1426861938 , <E> =  -2.88751719478 , dalpha(%) =  -0.405328380619
alpha =  0.142236192785 , <E> =  -2.89246890053 , dalpha(%) =  -0.315378105975
alpha =  0.141616653268 , <E> =  -2.88976900152 , dalpha(%) =  -0.43557093672
alpha =  0.141164316079 , <E> =  -2.89015535006 , dalpha(%) =  -0.319409601875
alpha =  0.140553038101 , <E> =  -2.89265316396 , dalpha(%) =  -0.433025849206
alpha =  0.139915751108 , <E> =  -2.89029624048 , dalpha(%) =  -0.453413886839
alpha =  0.139584640561 , <E> =  -2.89037146532 , dalpha(%) =  -0.236649944302
alpha =  0.139150092137 , <E> =  -2.89329648369 , dalpha(%) =  -0.311315358461
alpha =  0.13869915011 , <E> =  -2.88701990084 , dalpha(%) =  -0.324068794917
alpha =  0.138313539548 , <E> =  -2.8915495172 , dalpha(%) =  -0.27801941192
alpha =  0.137907774305 , <E> =  -2.8933591276 , dalpha(%) =  -0.293366249287
alpha =  0.137498359965 , <E> =  -2.8910343397 , dalpha(%) =  -0.296875460511
alpha =  0.137166571086 , <E> =  -2.8897800576 , dalpha(%) =  -0.241303880864
alpha =  0.136774668181 , <E> =  -2.89429082193 , dalpha(%) =  -0.285713131327
alpha =  0.136303685735 , <E> =  -2.89449652335 , dalpha(%) =  -0.344349178389
alpha =  0.135930951197 , <E> =  -2.88746809289 , dalpha(%) =  -0.273458883706
alpha =  0.135670767917 , <E> =  -2.88744155898 , dalpha(%) =  -0.191408415692
alpha =  0.135482886869 , <E> =  -2.89053299636 , dalpha(%) =  -0.138483072583
alpha =  0.135276733825 , <E> =  -2.888007238 , dalpha(%) =  -0.152161685192
alpha =  0.13489397768 , <E> =  -2.8882022753 , dalpha(%) =  -0.282943071187
alpha =  0.134495003174 , <E> =  -2.88817930338 , dalpha(%) =  -0.295768953595
alpha =  0.134185022363 , <E> =  -2.89140440099 , dalpha(%) =  -0.230477567016
alpha =  0.134065399586 , <E> =  -2.88996002645 , dalpha(%) =  -0.0891476373825
alpha =  0.133877583172 , <E> =  -2.89386193782 , dalpha(%) =  -0.140093129091
alpha =  0.133632445071 , <E> =  -2.88916537315 , dalpha(%) =  -0.183106159923
End result: alpha =  0.133338838698 in  200 iterations
Final Energy at alpha( 0.133338838698 ) = -2.87886359522 , var(E) =  0.118665415782
```

## Hydrogen Molecule

```python
>>> numbbeta = 10000
>>> beta = 0.55
>>> zeta = 0.51
>>> N = 400
>>> steps = 1000
>>> steps_final = 300000
>>> d = 2.0
>>> s = 1.4011
>>> nblocks = 10
...
>>> beta_old = beta
>>> dbeta = 1
>>> i = 0
...
>>> while abs(dbeta) > 1e-4 and i < numbbeta:
...     pos_walker = np.random.uniform(-2,2,(N,3,2,2))
...     Energy = np.zeros(shape=(steps,N))
...     lnpsi = np.zeros(shape=(steps,N))
...
...     Energy, lnpsi = simulate_hydrogen_molecule_min(s, beta, steps, pos_walker,N)
...     #meanEnNew = np.mean(Energy)
...     varE = np.var(Energy)
...     meanEn = np.mean(Energy)
...     #diffmeanEn = np.absolute(meanEnNew - meanEn)/abs(meanEn)
...     #meanEn = meanEnNew
...     print("beta = ",beta,", <E> = ", meanEn, "iteration = ", i, "dbeta(%) = ", dbeta*100)
...
...     meanlnpsi = np.mean(lnpsi)
...     meanEtimeslnpsi = np.mean(lnpsi*Energy)
...     dEdbeta = 2*(meanEtimeslnpsi-meanEn*meanlnpsi)
...     #beta -= ((i+1)**(-zeta))*dEdbeta
...     beta -= 0.5*dEdbeta
...     dbeta = (beta - beta_old)/beta_old
...     beta_old = beta
...     i += 1
...
>>> print("End result: beta = ",beta," in ", i,"iterations.")
...
>>> Energy_final = np.zeros(shape=(steps_final,))
>>> pos_walker = np.random.uniform(-2,2,(N,3,2,2))
>>> Energy_final = simulate_hydrogen_molecule_min(s, beta, steps_final, pos_walker, N)[0]
>>> Energy_truncated = Energy_final[7000:,:]
>>> varE_final = np.var(Energy_truncated)
>>> mean_error_calculated = Error(Energy_truncated,nblocks)[0]
>>> std_error_calculated = Error(Energy_truncated,nblocks)[1]
...
...
>>> print("mean with error function: ", mean_error_calculated, "and error: ", std_error_calculated)
beta =  0.55 , <E> =  -1.15069211947 iteration =  0 dbeta(%) =  100
beta =  0.551834076102 , <E> =  -1.1498029617 iteration =  1 dbeta(%) =  0.333468382219
beta =  0.553430780835 , <E> =  -1.14989067207 iteration =  2 dbeta(%) =  0.289345077118
beta =  0.554823478784 , <E> =  -1.14848020495 iteration =  3 dbeta(%) =  0.251648082609
beta =  0.556245075771 , <E> =  -1.14830143169 iteration =  4 dbeta(%) =  0.256225095357
beta =  0.557734767348 , <E> =  -1.15046171838 iteration =  5 dbeta(%) =  0.267812092416
beta =  0.559205893554 , <E> =  -1.15058943051 iteration =  6 dbeta(%) =  0.263768065449
beta =  0.560211302127 , <E> =  -1.15130477973 iteration =  7 dbeta(%) =  0.179792198975
beta =  0.561881443958 , <E> =  -1.14855945386 iteration =  8 dbeta(%) =  0.298127121872
beta =  0.562952299154 , <E> =  -1.14983022929 iteration =  9 dbeta(%) =  0.190583833615
beta =  0.56423637451 , <E> =  -1.14988470086 iteration =  10 dbeta(%) =  0.228096653617
beta =  0.565145702361 , <E> =  -1.15040600089 iteration =  11 dbeta(%) =  0.161160799247
beta =  0.565939690903 , <E> =  -1.14943179607 iteration =  12 dbeta(%) =  0.140492715168
beta =  0.567059412324 , <E> =  -1.15201860623 iteration =  13 dbeta(%) =  0.197851721547
beta =  0.568267010149 , <E> =  -1.14951324123 iteration =  14 dbeta(%) =  0.212957901557
beta =  0.569092283434 , <E> =  -1.14893590543 iteration =  15 dbeta(%) =  0.145226323121
beta =  0.569910896168 , <E> =  -1.1502839496 iteration =  16 dbeta(%) =  0.143845340755
beta =  0.570672030327 , <E> =  -1.15034451141 iteration =  17 dbeta(%) =  0.133553185997
beta =  0.571406828813 , <E> =  -1.14816402161 iteration =  18 dbeta(%) =  0.12876020676
beta =  0.572045335259 , <E> =  -1.14935541779 iteration =  19 dbeta(%) =  0.111742879768
beta =  0.572404324275 , <E> =  -1.14928147536 iteration =  20 dbeta(%) =  0.0627553435503
beta =  0.572874456113 , <E> =  -1.150498486 iteration =  21 dbeta(%) =  0.0821328243237
beta =  0.573683213679 , <E> =  -1.14968372726 iteration =  22 dbeta(%) =  0.141175358273
beta =  0.57415019344 , <E> =  -1.15002306836 iteration =  23 dbeta(%) =  0.0814002832958
beta =  0.575018852152 , <E> =  -1.15117366721 iteration =  24 dbeta(%) =  0.151294682447
beta =  0.575735854569 , <E> =  -1.14872906693 iteration =  25 dbeta(%) =  0.124691984416
beta =  0.576062066788 , <E> =  -1.15057659032 iteration =  26 dbeta(%) =  0.0566600492907
beta =  0.576531006264 , <E> =  -1.14976262064 iteration =  27 dbeta(%) =  0.0814043317303
beta =  0.576755728935 , <E> =  -1.15020839486 iteration =  28 dbeta(%) =  0.0389784190012
beta =  0.577170849825 , <E> =  -1.14764081407 iteration =  29 dbeta(%) =  0.0719751653134
beta =  0.577855332288 , <E> =  -1.14960820479 iteration =  30 dbeta(%) =  0.118592694596
beta =  0.578075553669 , <E> =  -1.15035214167 iteration =  31 dbeta(%) =  0.0381101235508
beta =  0.578668609274 , <E> =  -1.15177579829 iteration =  32 dbeta(%) =  0.1025913656
End result: beta =  0.578717425638  in  33 iterations.
mean with error function:  -1.15112092978 and error:  0.000304039949376
```

```python

```
