# my_geophysics_package_final

Pacote final integrado do projeto `gravmag_codes`.

## Objetivo

Este pacote reúne os módulos de modelagem e interpretação geofísica desenvolvidos no projeto, preservando os nomes das funções já usadas nos notebooks.

## Módulos principais

- `auxiliars.py` — funções auxiliares numéricas.
- `constants.py` — constantes físicas e conversões.
- `grids.py` — construção e manipulação de grids.
- `kernel.py` — kernels e funções de apoio para modelagem.
- `gravity.py` — funções gravimétricas gerais.
- `prism.py` — modelagem gravimétrica/magnética de prismas.
- `sphere.py` — modelagem de esferas.
- `cylinder.py` — modelagem de cilindros.
- `hexagonalprism.py` — prismas hexagonais.
- `polygonalprism.py` — prismas poligonais.
- `derivative.py` — derivadas e gradientes.
- `filtering.py` — filtros no domínio do espaço e frequência.
- `equivalentlayer.py` — camada equivalente.
- `regres.py` — regressão, ajuste polinomial, separação regional-residual, Lagrange e métricas.
- `statistical.py` — métricas estatísticas.
- `plotting.py` — rotinas de plotagem.
- `synthetic.py` — modelos e dados sintéticos.
- `inversion.py` — inversão geofísica e regularização.
- `seismic_methods.py` — métodos sísmicos sintéticos.
- `electrical_methods.py` — métodos elétricos e SEV.
- `electromagnetic_methods.py` — métodos eletromagnéticos, GPR e skin depth.
- `radiometry_welllog_methods.py` — radiometria K-U-Th e perfilagem de poço.
- `geomagnetism.py` — geomagnetismo, campo global, harmônicos esféricos sintéticos, RTP e mapas Cartopy.

## Como usar

Coloque a pasta `gravmag_codes` na mesma pasta do seu notebook e rode:

```python
from gravmag_codes import prism, filtering, derivative, regres
from gravmag_codes import geomagnetism as geomag
from gravmag_codes import radiometry_welllog_methods as rwl

print(regres.__file__)
print(geomag.__file__)
```

Para verificar módulos importados:

```python
from gravmag_codes import available_modules, import_errors
print(available_modules())
print(import_errors())
```

## Dependências recomendadas

```text
numpy
scipy
matplotlib
cartopy
pywavelets
lasio
```
