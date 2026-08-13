# -*- coding: utf-8 -*-
"""
gravmag_codes
=============

Pacote integrado de métodos geofísicos para modelagem, filtragem,
separação regional-residual, inversão, geomagnetismo, sísmica, métodos
elétricos/eletromagnéticos, radiometria e perfilagem de poço.

Observação: as funções públicas existentes foram preservadas.
"""

__version__ = "1.0.0"
__author__ = "Nelson Ribeiro Filho"

_EXPECTED_MODULES = ['constants', 'auxiliars', 'grids', 'kernel', 'gravity', 'prism', 'sphere', 'cylinder', 'hexagonalprism', 'polygonalprism', 'derivative', 'filtering', 'equivalentlayer', 'regres', 'statistical', 'plotting', 'synthetic', 'inversion', 'seismic_methods', 'electrical_methods', 'electromagnetic_methods', 'radiometry_welllog_methods', 'geomagnetism']
_IMPORTED_MODULES = []
_IMPORT_ERRORS = {}

def _safe_import(module_name):
    try:
        module = __import__(f"{__name__}.{module_name}", fromlist=[module_name])
        globals()[module_name] = module
        _IMPORTED_MODULES.append(module_name)
        return True
    except Exception as exc:
        _IMPORT_ERRORS[module_name] = repr(exc)
        return False

for _module_name in _EXPECTED_MODULES:
    _safe_import(_module_name)

def available_modules():
    """Return modules successfully imported from gravmag_codes."""
    return list(_IMPORTED_MODULES)

def expected_modules():
    """Return the complete expected module list."""
    return list(_EXPECTED_MODULES)

def import_errors():
    """Return optional import errors captured during package initialization."""
    return dict(_IMPORT_ERRORS)
