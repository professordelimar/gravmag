# -*- coding: utf-8 -*-
"""
Example 02 - Tilt angle from synthetic gravity anomaly.
"""

from gravmag_codes import synthetic, filtering, plotting

result = synthetic.my_synthetic_prism_gravity_test(
    area=[0, 10000, 0, 10000],
    shape=(101, 101),
    observation_level=0.0,
    top=[500.0, 800.0],
    bottom=[2500.0, 3000.0],
    dx=1000.0,
    dy=1000.0,
    density=[2.60, 2.85],
    noise_percent=0.0,
    seed=42
)

x = result["x"]
y = result["y"]
gz = result["gz_true"]

tilt = filtering.my_tilt(
    x, y, gz,
    pad_factor=0.5,
    pad_mode="reflect",
    taper=True,
    degrees=True,
    flatten=False
)

fig, ax, cf = plotting.my_contourf(
    x, y, tilt,
    xlabel="X (m)",
    ylabel="Y (m)",
    colorbar_label="Tilt angle (degree)",
    colormap="coolwarm",
    levels=50,
    center_zero=True,
    title="Tilt angle from synthetic gz"
)

plotting.my_contour(
    x, y, tilt,
    ax=ax,
    levels=[-45, 0, 45],
    colors="black",
    linewidths=0.8,
    clabel=True
)

plotting.my_save_figure("example_02_tilt_angle.png", fig=fig)
