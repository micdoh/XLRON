# Comparison with optical-networking-gym OSNR Model

This page compares XLRON's GN model with the OSNR computation in [optical-networking-gym](https://github.com/carlosnatalino/optical-networking-gym) (`optical_networking_gym.core.osnr` and `optical_networking_gym.envs.qrmsa`), highlighting the key differences in physics, implementation, and scope.

## Model Formulation

Both implementations compute **GSNR** (generalised signal-to-noise ratio) as the ratio of signal power to the sum of ASE noise and nonlinear interference, accumulated across spans on the path:

```
1/GSNR_path = sum over spans of (P_ASE + P_NLI) / P_signal
```

The NLI in both cases is based on the **GN model closed-form approximation**, but they differ in the level of detail:

| Aspect | optical-networking-gym | XLRON |
|---|---|---|
| **NLI formula** | Single `arcsinh`-based expression combining SPM and XPM into one `sum_phi` accumulator | Separate SPM (`eta_spm`) and XPM (`eta_xpm`) efficiency coefficients with distinct formulas |
| **SPM term** | `arcsinh(pi^2 * \|beta_2\| * B^2 / (4*alpha))` -- standard GN model form | `arcsinh`-based but includes ISRS Raman tilt correction term `T`, dispersion slope `beta_3`, and optionally `beta_4` |
| **XPM term** | `arcsinh(upper) - arcsinh(lower)` using channel frequency separation and bandwidth, minus a modulation-format-dependent correction | `arctan`-based XPM formula with full ISRS Raman tilt, inter-channel dispersion walk-off (`phi_ik` depends on `beta_2`, `beta_3`, `beta_4`), and optional modulation format excess kurtosis correction |
| **ISRS (Raman tilt)** | Not modelled | Full ISRS correction: Raman gain slope `C_r` modifies effective attenuation per channel, with tilt term `T` in both SPM and XPM |
| **Dispersion model** | Single coefficient `beta_2` only | Three coefficients: `beta_2`, `beta_3` (dispersion slope), `beta_4` (curvature) |

## ASE Noise

| Aspect | optical-networking-gym | XLRON |
|---|---|---|
| **Formula** | `P_ASE = B * h * f * (exp(2*alpha*L) - 1) * NF_linear` | `P_ASE = 2 * N_sp * (G-1) * h * f * B` where `N_sp` is derived from NF and gain |
| **Gain model** | Implicit: `G = exp(2*alpha*L)` (pure loss compensation, no ISRS) | ISRS-aware gain: `G = exp(alpha*L) / g_SRS` where `g_SRS` corrects for Raman tilt |
| **Per-band NF** | Single scalar NF per span (or single default for all spans) | Per-frequency-slot NF loaded from CSV, supporting multi-band scenarios (S/C/L) |
| **ROADM ASE** | Not modelled | Full ROADM model with configurable express and add/drop losses, booster amplifier NF |
| **DRA support** | Not available | Optional Friis cascade model for hybrid Raman+EDFA with per-channel Raman gain from ODE solution |

## NLI Computation Details

The optical-networking-gym NLI uses a simplified formula derived from the incoherent GN model (Poggiolini 2014):

```python
# SPM contribution (self-channel)
sum_phi = arcsinh(pi^2 * |beta_2| * B_ch^2 / (4*alpha))

# XPM contribution (each interfering channel k)
phi_k = arcsinh(pi^2 * |beta_2| * L_eff_a * B_k * (delta_f + B_k/2))
      - arcsinh(pi^2 * |beta_2| * L_eff_a * B_k * (delta_f - B_k/2))
      - phi_mod_k * (B_k / |delta_f|) * (5/3) * (L_eff / L)

P_NLI = (P_ch / B_ch)^3 * (8 / (27*pi*|beta_2|)) * gamma^2 * L_eff * sum_phi * B_ch
```

XLRON computes SPM and XPM separately with higher-order dispersion and ISRS:

```python
# SPM efficiency (per channel i)
eta_spm = (4/9) * (gamma^2/B^2) * pi / (phi_i * a_bar * (2a + a_bar))
          * [(T - a^2)/a * arcsinh(phi_i*B^2/(a*pi))
             + ((a+a_bar)^2 - T)/(a+a_bar) * arcsinh(phi_i*B^2/((a+a_bar)*pi))]

# XPM efficiency (channel i due to channel k)
eta_xpm_ik = (32/27) * (P_k/P_i)^2 * (gamma^2/denom)
              * [(T_k - a_k^2)/a_k * arctan(phi_ik*B_i/a_k)
                 + ((a_k+a_bar_k)^2 - T_k)/(a_k+a_bar_k) * arctan(phi_ik*B_i/(a_k+a_bar_k))]
```

Key physics differences:

- **ISRS tilt term `T`**: XLRON includes `T = (a + a_bar - f*P_total*Cr)^2` which captures the frequency-dependent power transfer due to stimulated Raman scattering. This is absent in optical-networking-gym.
- **`arctan` vs `arcsinh` for XPM**: XLRON uses `arctan` for XPM (appropriate for inter-channel walk-off), while optical-networking-gym uses `arcsinh` (the same kernel as SPM). The `arctan` form is the standard closed-form GN model result for XPM.
- **Modulation format correction**: optical-networking-gym applies a `phi_modulation_format` correction per interfering channel based on spectral efficiency index. XLRON uses per-format `excess_kurtosis` values in an optional correction term.
- **Coherent accumulation**: XLRON supports both coherent and incoherent NLI accumulation across spans (controlled by `--coherent` flag). optical-networking-gym assumes incoherent accumulation (linear scaling with span count).

## Fibre Parameters

| Parameter | optical-networking-gym | XLRON | Notes |
|---|---|---|---|
| `beta_2` | -21.3e-27 s^2/m (hardcoded) | Derived from `--dispersion_coeff` (default 17e-6 s/m^2) | XLRON value gives beta_2 ~ -21.7e-27 at 1550 nm |
| `gamma` | 1.3e-3 1/(W.m) (hardcoded) | `--nonlinear_coefficient` (default 1.2e-3) | Slightly different defaults |
| `alpha` | Per-span from topology, or single default | `--attenuation` (default 4.605e-5 Np/m = 0.2 dB/km) | XLRON also supports per-band values |
| `NF` | Per-span from topology, or single default | Per-frequency-slot from CSV, or `--amplifier_noise_figure` | Multi-band support in XLRON |
| Raman gain slope `C_r` | Not used | `--raman_gain_slope` (default 2.8e-17) | ISRS tilt correction |
| Dispersion slope | Not modelled | `--dispersion_slope` (default 60.7 s/m^3) | Higher-order dispersion |

## Environment Integration

| Aspect | optical-networking-gym (`QRMSAEnv`) | XLRON (`rmsa_gn_model` / `rsa_gn_model`) |
|---|---|---|
| **Language** | Cython (`.pyx`) compiled to C for speed | JAX (Python), JIT-compiled to XLA for GPU/TPU |
| **Execution** | Sequential: Python loops over links, spans, and interfering services | Vectorised: `jax.vmap` over links and channels; broadcasting for XPM N x N matrix |
| **When GSNR is computed** | At action time: `calculate_osnr()` called once per candidate placement | At masking time (rmsa) or post-placement (rsa): full link SNR array recomputed |
| **Disruption checking** | After provisioning, recalculates OSNR for all co-link services to detect disruptions | Built into action check: all existing channels on affected links must still meet thresholds |
| **Parallelism** | Single environment, single service at a time | `NUM_ENVS` parallel environments via batched vectorisation |
| **Topology** | NetworkX graph with per-edge attributes | Static JAX arrays (`link_slot_array`, `channel_power_array`, etc.) |
| **Action space** | `(path_index, modulation_index, initial_slot)` tuple | Flat integer encoding path and slot (modulation selected automatically in `rmsa_gn_model`) |
| **Reward** | `1 - log10(1 + OSNR - min_OSNR) / 1.6` (OSNR-margin-aware) | Binary: +1 for accepted, 0 for blocked (configurable via reward shaping) |

## Span Model

| Aspect | optical-networking-gym | XLRON |
|---|---|---|
| **Span definition** | Explicit `Span` objects per link with individual `length`, `attenuation_normalized`, `noise_figure_normalized` | Computed from link length: `num_spans = ceil(link_length / max_span_length)`, all spans equal length |
| **Per-span parameters** | Each span can have unique attenuation and NF | Uniform parameters per link (all spans share same fibre type) |
| **Multi-span links** | Explicit loop over topology-defined spans | Automatic subdivision of links into equal-length spans |

## Power Model

| Aspect | optical-networking-gym | XLRON |
|---|---|---|
| **Launch power** | Per-service, set at provisioning time from `launch_power_dbm` | Per-channel, configurable via `--power_per_channel` or RL-controlled |
| **Power budget** | No explicit per-fibre power constraint | `--max_power_per_fibre` enforced during masking and action checking |
| **Power spectral density** | `P_ch / B_ch` used as PSD in NLI formula | Per-channel power with ISRS-aware tilt across spectrum |

## Scope and Capabilities Summary

| Feature | optical-networking-gym | XLRON |
|---|---|---|
| GN model NLI (SPM + XPM) | Yes (simplified) | Yes (full ISRS GN model) |
| ISRS Raman tilt | No | Yes |
| Higher-order dispersion (beta_3, beta_4) | No | Yes |
| Coherent NLI accumulation | No | Optional |
| Modulation format NLI correction | Yes (phi array) | Optional (excess kurtosis) |
| Nyquist subchannel modelling | No | Yes (`--num_subchannels`) |
| ROADM noise model | No | Yes |
| Transceiver back-to-back noise | No | Yes |
| Per-band amplifier NF / TRX SNR | No | Yes (CSV data) |
| Distributed Raman Amplification | No | Yes (`--use_raman_amp`) |
| Multi-band (S+C+L) | No | Yes |
| Per-fibre power budget | No | Yes |
| GPU/TPU acceleration | No (CPU/Cython) | Yes (JAX JIT + vmap) |
| Parallel environments | No | Yes (`--NUM_ENVS`) |
| Differentiable through physics | No | Yes (`--differentiable`) |
| Service disruption tracking | Yes (recalculates OSNR for co-link services) | Yes (built into SNR validity check) |
