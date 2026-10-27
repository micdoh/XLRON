"""Independent NumPy audit of two formerly discrepant GN formulas.

Both have since been fixed in production: the generic XPM channel/interferer
axes, and the SPM bracket grouping (the Eq. (9) leading factor now multiplies
both arcsinh terms), so both relative differences should now be at float32
rounding level. This is a diagnostic, not a replacement propagation model. The
bracket grouping and channel axes follow the cited Semrau implementation; beta4
follows XLRON. No production formulas are changed by this audit. Run with:
  uv run python -m experimental.benchmarks.gn_accuracy_audit --output /tmp/audit.json
"""

import argparse
import json
from pathlib import Path

import jax.numpy as jnp
import numpy as np
from scipy.constants import c, pi

from xlron.environments.gn_model.isrs_gn_model import _spm, _xpm


def audit():
    f = np.linspace(-5e12, 5e12, 8)
    bw = np.linspace(25e9, 100e9, 8)
    power = np.linspace(0.3e-3, 1.3e-3, 8)
    a = abar = 0.2 / 4.343 / 1e3
    gamma, wavelength, dispersion, slope, cr = 1.2e-3, 1577.5e-9, 17e-6, 67.0, 2.8e-17
    beta2 = -dispersion * wavelength**2 / (2 * pi * c)
    beta3 = wavelength**3 / (2 * pi * c) ** 2 * (wavelength * slope + 2 * dispersion)
    beta4 = (
        -(wavelength**4)
        / (2 * pi * c) ** 3
        * (6 * wavelength * slope + 6 * dispersion + slope * wavelength**2)
    )
    phi = 1.5 * pi**2 * (beta2 + 2 * pi * beta3 * f + 2 * pi**2 * beta4 * f**2)
    tilt = (a + abar - f * power.sum() * cr) ** 2
    prefactor = 4 / 9 * gamma**2 / bw**2 * pi / (phi * abar * (2 * a + abar))
    spm_reference = prefactor * (
        (tilt - a**2) / a * np.arcsinh(phi * bw**2 / a / pi)
        + ((a + abar) ** 2 - tilt) / (a + abar) * np.arcsinh(phi * bw**2 / (a + abar) / pi)
    )
    spm_actual = np.asarray(
        _spm(jnp.asarray(phi), jnp.asarray(tilt), jnp.asarray(bw), a, abar, gamma)
    )

    fi, fk = f[:, None], f[None, :]
    phi_ik = (
        2
        * pi**2
        * (fi - fk)
        * (beta2 + pi * beta3 * (fi + fk) + 2 / 3 * pi**2 * beta4 * (fi**2 + fi * fk + fk**2))
    )
    denom = bw[None, :] * phi_ik * abar * (2 * a + abar)
    safe_denom = np.where(denom != 0, denom, 1)
    xpm_terms = (
        32
        / 27
        * (power[None, :] / power[:, None]) ** 2
        * gamma**2
        / safe_denom
        * (
            (tilt[None, :] - a**2) / a * np.arctan(phi_ik * bw[:, None] / a)
            + ((a + abar) ** 2 - tilt[None, :])
            / (a + abar)
            * np.arctan(phi_ik * bw[:, None] / (a + abar))
        )
    )
    xpm_reference = np.where(denom != 0, xpm_terms, 0).sum(axis=1)
    xpm_actual = np.asarray(
        _xpm(
            jnp.asarray(power),
            jnp.asarray(power),
            jnp.asarray(phi_ik),
            jnp.asarray(tilt[None, :]),
            jnp.asarray(bw),
            jnp.asarray(bw),
            a,
            abar,
            gamma,
        )
    )
    return {
        "scope": "Diagnostic coefficients only; not a full link-SNR or experimental validation",
        "reference": "https://github.com/dsemrau/ISRSGNmodel/blob/master/Functions/Python/ISRSGNmodel.py",
        "inputs": {
            "frequency_hz": f.tolist(),
            "bandwidth_hz": bw.tolist(),
            "power_w": power.tolist(),
        },
        "spm": {
            "implemented": spm_actual.tolist(),
            "bracketed_reference": spm_reference.tolist(),
            "max_relative_difference": float(
                np.max(abs(spm_actual - spm_reference) / abs(spm_reference))
            ),
        },
        "xpm": {
            "implemented_generic": xpm_actual.tolist(),
            "explicit_axes_reference": xpm_reference.tolist(),
            "max_relative_difference": float(
                np.max(abs(xpm_actual - xpm_reference) / abs(xpm_reference))
            ),
        },
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    result = json.dumps(audit(), indent=2) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(result)
    print(result)
