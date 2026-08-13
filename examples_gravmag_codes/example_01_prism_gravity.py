# -*- coding: utf-8 -*-
"""
Example 01 - Synthetic gravity anomaly from rectangular prisms.
"""

from gravmag_codes import synthetic, plotting

result = synthetic.my_synthetic_prism_gravity_test(
    area=[0, 10000, 0, 10000],
    shape=(101, 101),
    observation_level=0.0,
    top=[500.0, 800.0],
    bottom=[2500.0, 3000.0],
    dx=1000.0,
    dy=1000.0,
    density=[2.60, 2.85],
    noise_percent=1.0,
    seed=42
)

x = result["x"]
y = result["y"]
gz = result["gz_noisy"]

fig, ax, cf = plotting.my_contourf(
    x, y, gz,
    xlabel="X (m)",
    ylabel="Y (m)",
    colorbar_label="gz (mGal)",
    colormap="coolwarm",
    levels=50,
    title="Synthetic gravity anomaly"
)

plotting.my_draw_rectangle([2500, 3500, 4500, 5500], ax=ax, edgecolor="black", linestyle="--")
plotting.my_draw_rectangle([6000, 7000, 5000, 6000], ax=ax, edgecolor="black", linestyle="--")

plotting.my_save_figure("example_01_prism_gz.png", fig=fig)
