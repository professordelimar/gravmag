# -*- coding: utf-8 -*-
"""
Example 03 - Linear inversion with Tikhonov regularization.
"""

import numpy as np
from gravmag_codes import inversion

np.random.seed(42)

ndata = 80
nparams = 20

G = np.random.randn(ndata, nparams)

m_true = np.zeros(nparams)
m_true[5:10] = 2.0
m_true[12:15] = -1.0

d_true = G @ m_true
d_obs = d_true + np.random.normal(0.0, 0.05*np.std(d_true), size=ndata)

result = inversion.my_linear_inversion(
    G, d_obs,
    method="tikhonov",
    regulator=1.0e-2,
    regularization="first"
)

print("RMSE:", result["report"]["rmse"])
print("Correlation:", result["report"]["correlation"])
print("Estimated model:", result["model"])
