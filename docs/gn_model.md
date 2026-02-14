# GN Model Physical Layer

This page describes the physical layer model used by XLRON's GN model environments (`rsa_gn_model` and `rmsa_gn_model`). It covers the modelling assumptions, noise sources, how ROADM nodes are treated, and what each configuration parameter controls.

The implementation is based on the closed-form ISRS GN model from:

> D. Semrau, R. I. Killey, P. Bayvel, "A Closed-Form Approximation of the Gaussian Noise Model in the Presence of Inter-Channel Stimulated Raman Scattering," *J. Lightw. Technol.*, vol. 37, no. 9, pp. 1924-1936, May 2019.

## Overview of the Model

The GN model computes the signal-to-noise ratio (SNR) for each frequency slot on each fibre link in the network. The total noise on a channel is the sum of four independent contributions:

```
P_noise = P_ASE_inline + P_ASE_ROADM + P_NLI + P_TRX
```

| Noise source | Symbol | Origin |
|---|---|---|
| Inline amplifier ASE | P_ASE_inline | Erbium-doped fibre amplifier (EDFA) spontaneous emission at each span |
| ROADM ASE | P_ASE_ROADM | Booster amplifiers at express and add/drop ROADM nodes |
| Nonlinear interference | P_NLI | Kerr effect in the fibre (SPM + XPM, with ISRS correction) |
| Transceiver noise | P_TRX | Back-to-back transceiver SNR limit |

The per-channel SNR on a single link is then:

```
SNR_link = P_signal / P_noise
```

For a multi-link path, noise-to-signal ratios are summed across links (independent noise sources add in power):

```
1/SNR_path = sum over links on path of (1/SNR_link) + ROADM_ASE/P_signal
```

Note that inline ASE and NLI are computed **per link**, while ROADM ASE is computed **per path** (since it depends on the number of intermediate nodes traversed).


## Link Model: Fibre Spans and Inline Amplifiers

Each link in the topology is divided into fibre spans. Topology files store link distances in km; the environment converts these to metres and divides each link into equal-length spans:

```
num_spans = ceil(link_length_m / max_span_length)
span_length = link_length_m / num_spans
```

The default `max_span_length` is 100 km. For example, a 350 km link has 4 spans of 87.5 km each.

Each span has an inline EDFA at the output that compensates for fibre loss. The amplifier gain accounts for both fibre attenuation and the ISRS (Inter-channel Stimulated Raman Scattering) Raman tilt, which causes higher-frequency channels to transfer power to lower-frequency channels during propagation. The ISRS-aware gain per channel is:

```
G_i = exp(alpha * L) / g_SRS_i
```

where `g_SRS_i` corrects for the frequency-dependent power transfer due to Raman scattering.


### Fibre Attenuation

Fibre loss is characterised by the attenuation coefficient `alpha` in Nepers per metre. The default is 0.2 dB/km, which converts to approximately 4.605 x 10^-5 Np/m. The `attenuation_bar` parameter (mean attenuation, used internally for ISRS tilt calculations) defaults to the same value.


### Inline Amplifier ASE Noise

Each inline EDFA adds ASE noise according to:

```
P_ASE = 2 * N_sp * (G - 1) * h * f * B
```

where:

- `N_sp = (NF_lin * G) / (2 * (G - 1))` is the spontaneous emission factor
- `NF_lin = 10^(NF_dB / 10)` is the amplifier noise figure in linear units
- `G` is the ISRS-aware amplifier gain
- `h` is Planck's constant (6.626 x 10^-34 J.s)
- `f` is the absolute channel frequency in Hz
- `B` is the channel bandwidth in Hz
- The factor of 2 accounts for both polarisation modes

The total inline ASE on a link is the sum over all spans: `P_ASE_inline = num_spans * P_ASE_per_span`.

The amplifier noise figure can vary across the spectrum. XLRON supports per-slot NF values loaded from a CSV file that maps spectral bands to noise figure and transceiver SNR values (see [Spectral Band Data](#spectral-band-data) below).


### Nonlinear Interference (NLI)

NLI arises from the Kerr effect in the fibre and has two components:

- **SPM (Self-Phase Modulation)**: a channel interferes with itself
- **XPM (Cross-Phase Modulation)**: other channels create interference through inter-channel nonlinear mixing

The GN model computes NLI power as:

```
P_NLI_i = P_i^3 * eta_n_i
```

where `eta_n_i = eta_SPM_i + eta_XPM_i` is the NLI efficiency coefficient, which depends on fibre parameters (gamma, beta_2, beta_3, alpha), channel spacing, channel bandwidths, and channel powers.

**ISRS correction**: The model incorporates the effect of stimulated Raman scattering on NLI. The Raman gain slope `C_r` modifies the effective attenuation seen by each channel, which in turn affects NLI efficiency. This means that NLI is not just a function of local channel properties but also depends on the total power and frequency distribution of all channels on the link.

**Coherent vs incoherent accumulation**: NLI from successive spans can accumulate either coherently (phased) or incoherently (power-summed), controlled by the `--coherent` flag. The coherence factor epsilon determines the scaling:

- Incoherent (`--coherent=False`, default): SPM scales linearly with span count, XPM scales linearly
- Coherent (`--coherent=True`): SPM scales as `num_spans^(1 + epsilon)` where `epsilon > 0`, producing slightly higher NLI

In practice, incoherent accumulation is a reasonable approximation for most fibre types and is the default.

**Uniform vs non-uniform spans**: When `--uniform_spans=True` (default), all spans on a link are assumed to have equal length, enabling a fast closed-form computation. When `--uniform_spans=False`, a `jax.lax.scan` loop iterates over spans with potentially different lengths.


### Modulation Format Correction

When `--mod_format_correction=True`, the NLI calculation includes correction terms that account for the non-Gaussian statistics of specific modulation formats. Each format has an `excess_kurtosis` value (e.g., BPSK = -1, 16QAM = -0.68) that modifies the XPM contribution. Formats closer to Gaussian (kurtosis = -1) produce less NLI correction. This is only relevant for the `rmsa_gn_model` environment where modulation formats are explicitly tracked per channel.


### Transceiver Noise

The transceiver SNR represents the back-to-back performance limit of the transmitter and receiver, independent of fibre propagation. It is modelled as an additive noise term:

```
P_TRX = P_signal / SNR_TRX_linear
```

Like the amplifier noise figure, the transceiver SNR can vary across the spectrum via the per-band CSV data file.


## ROADM Node Model

ROADM (Reconfigurable Optical Add-Drop Multiplexer) nodes are modelled with separate express and add/drop loss parameters. Each ROADM contains a booster amplifier that compensates for the ROADM insertion loss, and this amplifier adds ASE noise.

For a lightpath traversing a path with `N_links` links:

- **Source node**: 1 add ROADM with loss `roadm_add_drop_loss` (default 8 dB)
- **Intermediate nodes**: `N_links - 1` express ROADMs with loss `roadm_express_loss` (default 5 dB) each
- **Destination node**: 1 drop ROADM with loss `roadm_add_drop_loss` (default 8 dB)
- **All ROADM amplifiers**: noise figure `roadm_noise_figure` (default 5 dB)

The ROADM ASE noise is computed per path (not per link) because the number of intermediate nodes depends on the end-to-end route. The formula follows the same ASE calculation as inline amplifiers, but using the ROADM loss as the gain to be compensated:

```
G_express = 10^(roadm_express_loss / 10)
G_add_drop = 10^(roadm_add_drop_loss / 10)

P_ROADM_ASE = N_intermediate * ASE(G_express, NF_roadm) + 2 * ASE(G_add_drop, NF_roadm)
```


## Power Budget Enforcement

The total optical power on each fibre link is constrained by `max_power_per_fibre` (default 21 dBm). When a new channel would cause the total power on any link along its path to exceed this limit, the action is rejected:

- **During masking**: candidate placements that would exceed the power limit are masked out
- **During action checking**: the power constraint is verified after tentative placement

The per-channel launch power is set by `power_per_channel` (in dBm). If not specified, it defaults to `max_power_per_fibre` divided equally among all slots. For example, with `max_power_per_fibre=21` dBm (125.9 mW) and `link_resources=100`, the default per-channel power is approximately 1.26 mW (1.0 dBm).


## Spectral Band Data

The amplifier noise figure and transceiver SNR can vary across the optical spectrum. XLRON loads per-band values from a CSV file (`transceiver_amplifier_data.csv`) that defines spectral bands with associated NF and transceiver SNR. Each frequency slot is assigned the values of the band it falls within.

The default data covers five bands spanning the partial S-band, C-band, and L-band (approximately 1485-1625 nm):

| Band | Wavelength range | Amplifier NF | Transceiver SNR |
|---|---|---|---|
| Partial S-band | 1485-1520 nm | 7.0 dB | 15.8 dB |
| S/C transition | 1520-1529 nm | 9.0 dB | 17.8 dB |
| C-band | 1529-1568 nm | 5.5 dB | 21.2 dB |
| L-band (main) | 1568-1608 nm | 6.0 dB | 21.2 dB |
| L-band (edge) | 1608-1625 nm | 9.0 dB | 17.1 dB |

The C-band has the best amplifier performance (lowest NF) and transceiver performance (highest SNR), reflecting the maturity of C-band EDFA technology.

Band boundaries for inter-band gap enforcement (`--enforce_band_gaps`) are defined in a separate file (`band_data.csv`), which specifies the standard optical bands (O, E, S, C, L, U) and their frequency ranges. This can be overridden with `--band_data_filepath`.

### Band Preference for Heuristic Slot Allocation

When using first-fit or last-fit heuristics with a GN model environment, the `--band_preference` flag controls the order in which optical bands are filled. By default, slots are allocated in raw index order (i.e. by frequency). With `--band_preference`, slots in the most-preferred band are exhausted before moving to the next band.

For example, `--band_preference=C,L` will fill C-band slots first, then L-band slots. This is useful for multi-band scenarios where operators want to prioritise certain bands (e.g. fill C-band before spilling into L-band).

```bash
python -m xlron.train.train \
  --env_type=rsa_gn_model \
  --topology_name=nsfnet_deeprmsa_directed \
  --link_resources=100 --k=5 --load=250 \
  --continuous_operation --ENV_WARMUP_STEPS=3000 \
  --TOTAL_TIMESTEPS=100000 --NUM_ENVS=1 \
  --EVAL_HEURISTIC --path_heuristic=ksp_ff \
  --band_preference=C,L
```

The preference string is a comma-separated list of band names (matching the `band_name` column in `band_data.csv`). Bands not listed are appended in CSV order after the specified ones. The flag affects both first-fit and last-fit heuristics; for last-fit, slots within each band are filled from the high end first, but bands are still tried in preference order.


## Modulation Formats

For `rmsa_gn_model`, modulation formats are loaded from a CSV file. Each format specifies:

| Column | Description |
|---|---|
| `name` | Format name (e.g., QPSK, 16QAM) |
| `maximum_length` | Maximum optical reach in km (used for path pruning) |
| `spectral_efficiency` | Bits per symbol per polarisation (determines required slots) |
| `minimum_osnr` | Minimum required SNR threshold in dB |
| `inband_xt` | Inband crosstalk tolerance in dB |
| `excess_kurtosis` | Excess kurtosis for modulation format NLI correction |

The number of slots required for a request is `ceil(requested_bandwidth / (slot_size * spectral_efficiency))`.

The SNR margin parameter (`--snr_margin`, default 1 dB) is added to the `minimum_osnr` threshold when checking whether a channel's SNR is sufficient.

Default formats for the GN model environments (from `modulations_gn_model.csv`):

| Format | SE (b/s/Hz) | SNR threshold | Excess kurtosis |
|---|---|---|---|
| BPSK | 1 | 12.6 dB | -1.00 |
| QPSK | 2 | 12.6 dB | -1.00 |
| 8QAM | 3 | 18.6 dB | -0.82 |
| 16QAM | 4 | 22.4 dB | -0.68 |
| 32QAM | 5 | 26.4 dB | -0.52 |
| 64QAM | 6 | 30.4 dB | -0.32 |


### Calculating SNR Thresholds from Spectral Efficiency

Instead of using pre-specified `minimum_osnr` values from the CSV, the GSNR threshold can be calculated analytically from the modulation order using the `--calc_minimum_osnr` flag. When enabled, the `minimum_osnr` column is overwritten with values computed from:

```
GSNR_th(m') = f(m') * erfc_inv(g(m', beta_FEC))
```

where `m' = spectral_efficiency` is the modulation level (so that `M = 2^m'`), `beta_FEC` is the pre-FEC BER target (`--beta_fec`, default 1.5e-2), and `erfc_inv` is the inverse complementary error function. The formula has three cases:

- **m' in {1, 2}** (BPSK, QPSK): `GSNR_th = m' * erfc_inv(2 * beta_FEC)`
- **m' = 3** (8QAM): `GSNR_th = (2(M-1)/3) * erfc_inv(1.5 * beta_FEC)`
- **m' in {4, 5, 6}** (16QAM, 32QAM, 64QAM): `GSNR_th = (2(M-1)/3) * erfc_inv(m' * beta_FEC / (2(1 - 1/sqrt(M))))`

The result is converted to dB: `GSNR_th_dB = 10 * log10(GSNR_th)`.

With the default `beta_fec=1.5e-2`, the calculated thresholds are:

| Format | SE | Calculated GSNR threshold |
|---|---|---|
| BPSK | 1 | 1.9 dB |
| QPSK | 2 | 4.9 dB |
| 8QAM | 3 | 8.8 dB |
| 16QAM | 4 | 11.6 dB |
| 32QAM | 5 | 14.7 dB |
| 64QAM | 6 | 17.6 dB |


## Path-Level SNR Computation

The per-link SNR values (computed as described above) are combined into a path-level SNR for each frequency slot:

1. For each link on the path, convert link SNR to NSR (noise-to-signal ratio): `NSR_link = 1 / SNR_link`
2. Sum NSRs across all links on the path (independent noise sources add in power)
3. Add ROADM ASE contribution as `P_ROADM_ASE / P_signal` for occupied slots
4. Convert back to SNR in dB: `SNR_path_dB = 10 * log10(1 / NSR_total)`

This path-level SNR is what gets compared against modulation format thresholds.

---

## `rsa_gn_model` Environment

The `rsa_gn_model` environment performs Routing and Spectrum Assignment with the GN model providing physical layer awareness, but without per-step modulation-format-aware masking.

### Intended Use

- **Throughput capacity studies**: measure network-level Shannon throughput under realistic physical layer constraints
- **Fast simulations**: action masking uses standard RSA slot-availability checks (no GN model evaluation during masking), making it significantly faster than `rmsa_gn_model`
- **Scenarios with predetermined modulation**: when modulation format is fixed or determined by path distance rather than real-time SNR

### How It Works

1. **Action masking**: uses the standard RSA mask (contiguous free-slot check). No GN model evaluation during masking.
2. **Action execution**: places the lightpath and updates `channel_power_array`, `channel_centre_bw_array`, and `path_index_array` on the affected links.
3. **SNR update**: after each step, `link_snr_array` is recomputed for all links using the GN model.
4. **Action check**: `check_action_rmsa_gn_model` verifies that all active lightpaths still meet a basic SNR threshold and that the power budget is not exceeded. If the check fails, the placement is rolled back.
5. **Throughput measurement** (optional, with `--monitor_active_lightpaths`): at episode end, computes Shannon-Hartley throughput for all active lightpaths:

```
throughput_per_LP = log2(1 + SNR_linear) * slot_size_GHz * 2 * (1 - FEC_overhead)
```

The factor of 2 is for dual polarisation. The default FEC overhead is 28% (`--fec_threshold=0.28`).

### Observation Space

The observation includes the request (source, destination, bandwidth, holding time) plus per-path statistics:

- Mean free block size and free slot count
- Path length in hops and distance
- Number of active connections, mean power, and mean SNR of connections on each path

### When to Use

Use `rsa_gn_model` when you want physically-aware throughput measurement but don't need the GN model to be evaluated at every masking step. This is appropriate for studying how routing and spectrum assignment strategies affect achievable throughput, or for large-scale simulations where the per-step cost of full GN model masking would be prohibitive.


---

## `rmsa_gn_model` Environment

The `rmsa_gn_model` environment performs Routing, Modulation and Spectrum Assignment with full GN model evaluation during action masking. This is the most physically realistic environment in XLRON.

### Intended Use

- **Physically realistic RL training**: the agent sees only genuinely feasible actions, learning to make decisions that respect nonlinear interference constraints
- **Modulation-adaptive networking**: the environment automatically selects the best modulation format for each placement based on current network conditions
- **Studying NLI-aware resource allocation**: understanding how channel placement affects neighbouring channels through nonlinear interference

### How It Works

1. **Action masking** (the key difference from `rsa_gn_model`): for each candidate combination of (path, modulation format, slot position), the mask function:
    - Tentatively places the channel with the candidate's power and bandwidth
    - Runs the full ISRS GN model to recompute SNR across all affected links
    - Checks that the new channel meets its modulation format's SNR threshold
    - Checks that all existing channels still meet their respective thresholds
    - Checks that the total power on each link does not exceed `max_power_per_fibre`
    - Only marks the action as valid if all checks pass

2. **Candidate evaluation strategy**: to keep the candidate count manageable, only the first-fit (FF) and last-fit (LF) slot positions are evaluated per (path, modulation format) pair, giving `2 * k * M` candidates total (where `k` = number of paths, `M` = number of modulation formats). All candidates are evaluated in parallel using `jax.vmap`.

3. **Modulation format selection**: the mask stores the winning modulation format index for each valid slot position in `mod_format_mask`. When an action is taken, the environment looks up the pre-computed modulation format rather than re-evaluating.

4. **Action execution**: places the lightpath with the selected modulation format's spectral efficiency (determining slot count) and the configured launch power. Updates all per-slot state arrays.

5. **Action check**: same as `rsa_gn_model` -- verifies SNR sufficiency, RSA validity, and power budget.

### Performance Considerations

The masking step is computationally expensive because it runs the GN model for every candidate. Two optimisations are available:

- **`get_snr_link_array_fused`** (used when `--uniform_spans=True` and `--mod_format_correction=False`): a fully inlined version that reduces XLA operations by ~41-48% compared to the standard version
- **FF/LF only**: evaluating only first-fit and last-fit positions (rather than all free slots) keeps the candidate count at `O(k * M)` rather than `O(k * M * link_resources)`

### FEC Code Rate

The `--fec_rate` parameter (default 0.8) models the overhead introduced by forward error correction. When a request is successfully accepted, the bitrate counted towards `accepted_bitrate` is scaled by this factor:

```
accepted_bitrate += requested_datarate * fec_rate
```

This reflects the fact that a fraction `(1 - fec_rate)` of the transmitted symbols carry FEC redundancy rather than user data. For example, with `fec_rate=0.8`, a 100 Gbit/s request contributes 80 Gbit/s of effective user throughput to the `accepted_bitrate` metric.

This parameter only applies to `rmsa_gn_model`.

### Metrics

The `rmsa_gn_model` environment tracks:

- `accepted_services`: count of successfully routed requests
- `accepted_bitrate`: cumulative effective bandwidth of accepted requests (scaled by `fec_rate`)
- Blocking probability: fraction of requests that could not be served

It does not compute Shannon throughput (unlike `rsa_gn_model` with `--monitor_active_lightpaths`).


---

## Configuration Parameters

### Fibre Parameters

| Flag | Default | Units | Description |
|---|---|---|---|
| `--attenuation` | 4.605e-5 | Np/m | Fibre attenuation (0.2 dB/km) |
| `--nonlinear_coefficient` | 1.2e-3 | 1/(W.m) | Kerr nonlinear coefficient |
| `--dispersion_coeff` | 17e-6 | s/m^2 | Group velocity dispersion D (17 ps/nm/km) |
| `--dispersion_slope` | 60.7 | s/m^3 | Dispersion slope dD/dlambda (0.067 ps/nm^2/km) |
| `--raman_gain_slope` | 2.8e-17 | 1/(W.m.Hz) | ISRS Raman gain slope |
| `--ref_lambda` | 1564e-9 | m | Reference wavelength (C+L band centre) |
| `--max_span_length` | 100000 | m | Maximum span length (100 km) |
| `--coherent` | False | -- | Coherent NLI accumulation across spans |
| `--uniform_spans` | True | -- | Assume equal-length spans per link |

### ROADM Parameters

| Flag | Default | Units | Description |
|---|---|---|---|
| `--roadm_express_loss` | 5.0 | dB | Insertion loss of express (pass-through) ROADM |
| `--roadm_add_drop_loss` | 8.0 | dB | Insertion loss of add/drop ROADM |
| `--roadm_noise_figure` | 5.0 | dB | Noise figure of ROADM booster amplifiers |

### Power Parameters

| Flag | Default | Units | Description |
|---|---|---|---|
| `--max_power_per_fibre` | 21.0 | dBm | Maximum total launch power per fibre link |
| `--power_per_channel` | None | dBm | Per-channel launch power. If None, defaults to `max_power_per_fibre / link_resources` |
| `--launch_power_type` | fixed | -- | Power assignment strategy: `fixed`, `tabular`, `rl`, or `scaled` |

### SNR and Modulation Parameters

| Flag | Default | Units | Description |
|---|---|---|---|
| `--snr_margin` | 1 | dB | Margin added to modulation format SNR thresholds |
| `--mod_format_correction` | False | -- | Enable modulation-format-dependent NLI correction |
| `--modulations_csv_filepath` | (built-in) | -- | Path to modulation formats CSV file |
| `--calc_minimum_osnr` | False | -- | Calculate `minimum_osnr` from spectral efficiency using GSNR threshold formula (ignores CSV values) |
| `--beta_fec` | 1.5e-2 | -- | Pre-FEC BER target for GSNR threshold calculation (used with `--calc_minimum_osnr`) |
| `--fec_rate` | 0.8 | -- | FEC code rate applied to accepted bitrate in `rmsa_gn_model` (`effective_bitrate = requested * fec_rate`) |
| `--fec_threshold` | 0.28 | -- | FEC overhead fraction (28%) for throughput calculation in `rsa_gn_model` |
| `--max_snr` | 50.0 | dB | Upper SNR clamp for observations |
| `--min_snr` | 7.0 | dB | Lower SNR limit for throughput calculation |

### Spectral Parameters

| Flag | Default | Units | Description |
|---|---|---|---|
| `--slot_size` | 12.5 | GHz | Spectral width of each frequency slot |
| `--link_resources` | -- | -- | Number of frequency slots per link |
| `--guardband` | 1 | slots | Guard band between adjacent channels |
| `--enforce_band_gaps` | False | -- | Mark inter-band gap slots as unusable (from `band_data.csv`) |
| `--band_data_filepath` | None | -- | Path to band definition CSV (defaults to built-in `band_data.csv`) |
| `--band_preference` | None | -- | Comma-separated band fill order for first-fit/last-fit (e.g. `C,L,S`) |


## Summary: `rsa_gn_model` vs `rmsa_gn_model`

| Aspect | `rsa_gn_model` | `rmsa_gn_model` |
|---|---|---|
| Modulation format | Fixed (implicit) | Per-channel, SNR-adaptive |
| Action masking | Standard RSA (free-slot check) | Full GN model evaluation per candidate |
| Masking speed | Fast | Slow (GN model per candidate) |
| SNR check on action | Yes (post-placement) | Yes (post-placement) |
| Power budget check | Yes | Yes |
| Throughput computation | Shannon-Hartley (optional) | Not available |
| Key metric | Throughput (Gbit/s) | Blocking probability, accepted bitrate |
| Best for | Capacity studies, fast evaluation | Realistic RL training, NLI-aware allocation |
| Modulation format correction | Not applicable | Optional (`--mod_format_correction`) |
