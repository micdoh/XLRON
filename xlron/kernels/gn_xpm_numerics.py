"""Algebraic frequency rescaling for the experimental coefficient accuracy audit.

Float32 cannot represent SI beta4 values around 1e-55. CPU execution also flushes
subnormal beta3 values around 1e-40 to zero. Apply frequency-unit factors before
these coefficients enter a float32 computation; output XPM units are unchanged.
"""


def scaled_xpm_inputs(frequency, physics):
    """Return binary-scaled frequencies and corresponding XPM coefficients.

    Coefficients must be Python scalars or otherwise unrounded values here;
    rescaling a beta4 already rounded to float32 zero cannot recover it.
    """
    # A power of two preserves the original frequency mantissas exactly. A
    # decimal THz conversion can perturb close frequency differences by rounding.
    scale = 2.0**40
    coefficients = dict(physics)
    coefficients["beta2"] *= scale
    coefficients["beta3"] *= scale**2
    coefficients["beta4"] *= scale**3
    coefficients["cr"] *= scale
    return frequency / scale, coefficients
