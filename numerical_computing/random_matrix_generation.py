import numpy as np

rg = np.random.default_rng(33)

### floats
M = rg.random(size=(3, 3))
M

### integers
M = rg.integers(low=0, high=11, size=(3, 3))
M

### gaussian
M = rg.normal(0, 1, size=(3, 3))
M