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
>>> def simulate_hydrogen_atom(alpha,steps,x):
...     for j in range(steps):
...         x_new = x + (np.random.random_sample(np.shape(x)) - 0.5)*d
...         p = (np.exp(-alpha*np.linalg.norm(x_new, axis=1)) / np.exp(-alpha*np.linalg.norm(x, axis=1)))**2
...         m = (p > np.random.rand(N)).reshape(-1,1)
...         x = x_new*m + x*~m
...         Energy[j,:] = -1/np.linalg.norm(x, axis=1) -alpha/2*(alpha - 2/np.linalg.norm(x, axis=1))
...     return Energy
```

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
alpha =  0.4 , <E> =  0.511502893599 var(E) =  0.0232279719021
alpha =  0.45 , <E> =  0.49732808975 var(E) =  0.00459394580581
alpha =  0.5 , <E> =  0.5 var(E) =  0.0
alpha =  0.55 , <E> =  0.500323389598 var(E) =  0.00484030022212
alpha =  0.6 , <E> =  0.507413313966 var(E) =  0.0164912155162
alpha =  0.4 , <E> =  0.513340132625 var(E) =  0.0246575526699
alpha =  0.45 , <E> =  0.502316803676 var(E) =  0.00505310089505
alpha =  0.5 , <E> =  0.5 var(E) =  0.0
alpha =  0.55 , <E> =  0.502211960073 var(E) =  0.00476038413058
alpha =  0.6 , <E> =  0.508117981602 var(E) =  0.0162377248355
alpha =  0.4 , <E> =  0.51216479253 var(E) =  0.0247518593422
alpha =  0.45 , <E> =  0.501156624548 var(E) =  0.00521123753389
alpha =  0.5 , <E> =  0.5 var(E) =  0.0
alpha =  0.55 , <E> =  0.504648579184 var(E) =  0.00396893582004
alpha =  0.6 , <E> =  0.512587646956 var(E) =  0.0158100539541
alpha =  0.4 , <E> =  0.512329306282 var(E) =  0.0254617663193
alpha =  0.45 , <E> =  0.503364945179 var(E) =  0.00589897128806
alpha =  0.5 , <E> =  0.5 var(E) =  0.0
alpha =  0.55 , <E> =  0.501293989157 var(E) =  0.00499843646633
alpha =  0.6 , <E> =  0.507777297677 var(E) =  0.0171293536648
1 loop, best of 3: 15 s per loop
```

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
alpha =  0.8 , <E> =  -0.502729403287 var(E) =  0.0312039157114
alpha =  0.9 , <E> =  -0.504091387593 var(E) =  0.00963084966431
alpha =  1 , <E> =  -0.5 var(E) =  0.0
alpha =  1.1 , <E> =  -0.48878256309 var(E) =  0.0131105393633
alpha =  1.2 , <E> =  -0.480386474339 var(E) =  0.0612092874519
alpha =  0.8 , <E> =  -0.498999590403 var(E) =  0.0302438891935
alpha =  0.9 , <E> =  -0.502110849119 var(E) =  0.0085152364342
alpha =  1 , <E> =  -0.5 var(E) =  0.0
alpha =  1.1 , <E> =  -0.488450656855 var(E) =  0.0129008658865
alpha =  1.2 , <E> =  -0.473329707808 var(E) =  0.0593016067611
alpha =  0.8 , <E> =  -0.501644345488 var(E) =  0.0283364811087
alpha =  0.9 , <E> =  -0.503521771605 var(E) =  0.00939705018142
alpha =  1 , <E> =  -0.5 var(E) =  0.0
alpha =  1.1 , <E> =  -0.49073607115 var(E) =  0.0122485731201
alpha =  1.2 , <E> =  -0.469335257406 var(E) =  0.0611486097022
alpha =  0.8 , <E> =  -0.49668373101 var(E) =  0.0300870049811
alpha =  0.9 , <E> =  -0.50106700876 var(E) =  0.00876235001123
alpha =  1 , <E> =  -0.5 var(E) =  0.0
alpha =  1.1 , <E> =  -0.489955447523 var(E) =  0.0133068610301
alpha =  1.2 , <E> =  -0.474337217431 var(E) =  0.0570752910415
1 loop, best of 3: 47.2 s per loop
```

```python

```
