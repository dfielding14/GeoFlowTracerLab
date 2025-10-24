# Structure Functions: Approach and Pair Selection

This note explains how structure functions are computed in this codebase, with a focus on how pairs of points are selected and aggregated. It refers to concrete implementation points so you can trace logic quickly.

## Overview

- Goal: estimate isotropic pth‑order structure functions S_p(ℓ) for either a scalar field θ(x, y) or a 2‑D vector field u = (u_x, u_y).
- Core entry point: `structure_functions` for scalars or vectors (see scalar_advection/structure.py:250).
- Auxiliary: second‑order S_2 can also be computed via FFT‑based correlation for higher accuracy/speed (scalar_advection/structure.py:70, scalar_advection/structure.py:79).

At a high level, the method samples a set of discrete displacement vectors Δ = (dx, dy) grouped into separation (radius) bins, forms increments by periodic shifts, averages over all pairs for each Δ, then averages across all Δ that land in the same separation bin.

## Separation Bins (ℓ bins)

- Bin edges are integer‑valued and approximately log‑spaced between `r_min` and `r_max` (inclusive of edges), found by `find_ell_bin_edges` (scalar_advection/binning.py:12).
- Defaults: `r_min = 1`, `r_max = min(nx, ny)/2` (half the domain extent in pixels), `n_ell_bins = 40`.
- The routine searches for a geometric spacing that remains distinct after rounding to integers to ensure exactly `n_ell_bins + 1` unique edges (scalar_advection/binning.py:16).

## Displacement Sampling (how Δ are chosen)

- Function: `generate_displacements(ell_bin_edges, n_per_bin, seed, ndim=2)` (scalar_advection/structure.py:32).
- For each ℓ bin [ℓ_i, ℓ_{i+1}):
  - Sample `n_per_bin` radii geometrically between the bin’s start/stop (scalar_advection/structure.py:35–37).
  - Sample angles uniformly on [0, π] (not 2π); because increments with Δ and −Δ are equivalent for isotropic statistics, sampling a half‑plane suffices (scalar_advection/structure.py:38).
  - Quantize to integer pixel steps: `dx = round(r cos θ)`, `dy = round(r sin θ)` (scalar_advection/structure.py:39–41).
- Deduplication and canonical orientation:
  - Remove duplicate Δ caused by rounding (scalar_advection/structure.py:42).
  - Map all vectors to a canonical half‑plane to avoid double counting opposite directions: if `dx < 0` or (`dx == 0` and `dy < 0`), flip the sign (scalar_advection/structure.py:43–45).
  - Sort by radius (scalar_advection/structure.py:46–47).
- The total number of candidate displacements is controlled by `n_disp_total`; per‑bin sampling is `n_per_bin = max(1, n_disp_total // n_ell_bins)` (scalar_advection/structure.py:264–265). After rounding/uniquing, the actual number per bin may be lower; this is tracked and used for averaging.

## From Displacements to Pairs of Points

- For each valid Δ = (dx, dy) with radius r > 0, we consider all periodic pairs on the grid:
  - For every pixel index (j, i), pair it with `(j + dy) mod ny, (i + dx) mod nx`.
  - This is implemented efficiently either via `np.roll` (NumPy fallback) or via explicit modulo indexing inside Numba kernels.
  - NumPy fallback examples:
    - Scalar increments: `diff = roll(field, (dy, dx)) - field` (scalar_advection/structure.py:108–109).
    - Vector increments: compute `dux, duy` via rolls, then project to longitudinal/transverse components (scalar_advection/structure.py:137–141).
  - Numba kernels compute the same using periodic modulo (scalar_advection/structure.py:176–210 and scalar_advection/structure.py:212–247).

This means for structure functions every displacement uses all grid points (full pair set), not a subsample. The Monte‑Carlo style subsampling appears only in the Yaglom helper in examples (see below).

### 3D specifics

- For 3D, directions are sampled uniformly on the sphere S² using normal deviates; integer rounding produces (dx, dy, dz) triples. We canonicalize to a single half‑space by flipping signs so that (dx > 0) or (dx == 0 and dy > 0) or (dx == 0 and dy == 0 and dz ≥ 0). Deduplication follows rounding and canonicalization.
- Separation is r = sqrt(dx² + dy² + dz²). The rest of the pipeline (bin assignment, averaging) is unchanged.

## Binning and Averaging

- Assign each sampled Δ to a separation bin using its radius and the precomputed edges (scalar_advection/structure.py:268–275, scalar_advection/structure.py:273).
- For each Δ, compute the spatial mean of |δq|^p (scalar) or of the chosen vector increment component raised to p (vector), where δq is the increment from periodic shifting.
- For each ℓ bin, average these per‑Δ means over all Δ that fell into the bin; the `counts` vector tracks how many distinct Δ landed in each bin (scalar_advection/structure.py:286–297 and scalar_advection/structure.py:307–315). Final curves divide by `max(1, counts)` to avoid division by zero.

## Vector Components

- For vectors (u_x, u_y) we compute:
  - Magnitude increments: |δu| = sqrt((δu_x)^2 + (δu_y)^2).
- Longitudinal: δu · ê, where ê = Δ / |Δ| (scalar_advection/structure.py:273–275; projection at scalar_advection/structure.py:139–141).
- Transverse, 2‑D: scalar transverse component −δu_x ê_y + δu_y ê_x (absolute value is the perpendicular magnitude).
- Transverse, 3‑D: magnitude of the component perpendicular to ê, i.e. |δu⊥| = sqrt(|δu|² − (δu·ê)²).
- Longitudinal increments can be treated as signed if `signed_longitudinal=True` (integer orders required), otherwise we use absolute values (scalar_advection/structure.py:155–161, scalar_advection/structure.py:236–247).

## Optional: Exact S_2 via FFT

- For p = 2, we can replace the Monte‑Carlo average by the exact isotropic S_2 using Wiener–Khinchin:
  - Compute autocorrelation via inverse FFT of |FFT(field)|^2, then `S2 = 2(⟨q^2⟩ − C(r))`.
  - Implementations: scalar field (scalar_advection/structure.py:70–76) and vector field (scalar_advection/structure.py:79–86), followed by radial binning with `radial_profile_from_map` (scalar_advection/structure.py:55–67).
- The main driver swaps in the FFT‑based S_2 for any requested order exactly equal to 2 (within tolerance) in 2‑D. For 3‑D the code currently uses the displacement‑based estimator only (FFT radial profiling helper is 2‑D).

## Reproducibility and Parameters

- `n_ell_bins`: number of separation bins. Larger gives finer radial resolution.
- `n_disp_total`: total target number of displacement samples across all bins; per‑bin target is `≈ n_disp_total / n_ell_bins`.
- `seed`: RNG seed that affects the random angles used when generating displacements; after integer rounding and deduplication, the exact set of Δ can vary with the seed.
- `r_min`, `r_max`: minimum/maximum separations. Default `r_max = min(nx, ny)/2` to avoid wrapping shorter than half the box.

Notes:
- Because of integer rounding and deduplication, bins can receive different counts of Δ; normalization uses the per‑bin `counts` to produce unbiased bin means.
- The periodic domain assumption is built into the implementation via roll/modulo.

## Pair‑increment PDFs (related diagnostic)

- `pair_increment_pdf` builds PDFs of increment amplitudes vs separation using the same Δ sampling and periodic pairing (scalar_advection/structure.py:502–627).
- With uniform `du` bin widths and Numba available, a specialized kernel accelerates counting (scalar_advection/structure.py:454–499, scalar_advection/structure.py:568–586).
- Structure functions can then be recovered as moments of the PDF (`structure_functions_from_pair_pdf`, scalar_advection/structure.py:724–755).

## Yaglom‑like Mixed Statistic in Examples

- The example driver computes ⟨|δu| |δθ|²⟩ similarly but, for cost control, draws a random subset of spatial pairs per Δ (instead of all) (examples/run_wavelet_scalar_experiment.py:508–560). Displacements themselves are still generated by `generate_displacements` (examples/run_wavelet_scalar_experiment.py:521).

## API Pointers

- High‑level accessors on the façade: `ScalarAdvectionAPI.scalar_structure_functions`, `ScalarAdvectionAPI.velocity_structure_functions` (scalar_advection/api.py:53–76).
- Plotting helper to visualize `(S_p)^{1/p}` and annotate power‑law fits: `plot_structure_functions` (scalar_advection/structure.py:338–420).

## Practical Tips

- Increase `n_disp_total` for smoother curves, especially at higher orders.
- Keep `n_ell_bins` modest (e.g., 32–48) unless you raise `n_disp_total` accordingly.
- Use FFT‑based S_2 for accurate second‑order baselines; it’s automatic by default (`use_fft_for_p2=True`).
