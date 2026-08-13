import numpy as np

def test_regres_polynomial_shape():
    from gravmag_codes import regres
    xv = np.linspace(0, 10, 11)
    yv = np.linspace(0, 10, 11)
    x, y = np.meshgrid(xv, yv)
    data = 1 + 2*x + 3*y
    result = regres.my_remove_polynomial_regional_2d(x=x, y=y, data=data, degree=1, full=True, normalize=True)
    assert result["regional"].shape == data.shape
    assert result["residual"].shape == data.shape
    assert np.all(np.isfinite(result["regional"]))
