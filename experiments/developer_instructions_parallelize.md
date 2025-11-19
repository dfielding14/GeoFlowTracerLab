**Task:** Create a new driver script that reproduces the functionality of the existing `Run_Wavelet_Scalar_Experiment.py` (same overall behavior, inputs, and outputs), but redesigned so that *each* ((\alpha, \kappa)) pair runs on its own node. The goal is to run on a supercomputer where each node has 32 cores, treating each ((\alpha, \kappa)) simulation as an embarrassingly parallel job, enabling higher resolution and faster wall-clock turnaround.

> **Note:** Shared plotting/snapshot helpers that were previously defined inside other `experiments/*.py` scripts are now copied into `scalar_advection/analysis_utils.py`. Future experiment scripts should only import from the `scalar_advection` package (not from other `experiments` modules) so that these utilities remain available even if legacy experiment files are removed.

---

### 1. Simulation script: `runSingleAlphaKappa.py`

**Purpose:**
Run a single passive scalar simulation for a given (\alpha) and (\kappa), evolve the scalar with a fixed velocity field, and stream snapshots/metadata to disk so that no history of scalar fields remains in memory.

**Inputs (via CLI args or config):**

At minimum:

* `alpha` (float)
* `kappa` (float)
* `t_end` (float): final simulation time
* `n_save` (int): target number of scalar snapshots between (t=0) and (t_{\rm end})
* Optional `dt_save` (float): fixed time between outputs (overrides `n_save`)
* `velocity_urms` (float, default 1.0): target RMS speed for the generated velocity (rescaled unless set ≤ 0)
* Time step / CFL or equivalent integration parameters
* Grid size and domain parameters
* Seed(s) or parameters required to construct the velocity field
* Boolean flag: `use_mean_gradient` (True/False) for whether the scalar has a mean gradient (implement the plumbing, comment in code if exact physics choice is TBD)
* Output base directory
* Any other required hyperparameters already used in the existing codebase (match existing style).
* **Restart controls:** `--restart-run` (path to an existing run directory), `--restart-snapshot` (explicit theta snapshot), `--restart-time` (physical time if it cannot be inferred), and `--velocity-file` (precomputed velocity field). These make it possible to load the saved velocity and the most recent (or any user-specified) scalar snapshot so that a long production run can be resumed if it hits the wall-clock limit.

**Core behavior:**

1. Construct the velocity field corresponding to the specified `alpha` and `kappa`.

   * Velocity field is **fixed in time** during the evolution.
   * Reuse existing utilities/code patterns for constructing the velocity field if available.
2. Initialize the scalar field (respecting the `use_mean_gradient` flag).
3. Evolve the scalar field from (t=0) to `t_end` with the given velocity field.
4. Save during integration:

   * Velocity field (`velocity_fields.npz`) and a quick-look magnitude plot.
   * Scalar snapshots written immediately when an output time is reached. Each snapshot is stored as `.npz` (`theta` array only) plus a manifest recording the save times. No list of past scalars is kept in RAM.
* As soon as each snapshot appears, a worker thread writes the corresponding theta/velocity overlay PNG (`analysis/theta_velocity_frames/theta_u_t*.png`), so movie frames accumulate while the run is still integrating. When a run is restarted the solver loads the specified snapshot, sets `t_start` to its physical time (parsed from the filename/manifest or from `--restart-time`), and continues stepping and writing snapshots at absolute times, so the new manifest always reflects the full physical timeline.
   * Dissipation time series (using solver diagnostics) can still be written directly since it does not require storing full fields.

---

### 2. Analysis script: `analyzeAlphaKappaRun.py`

After a run is finished, execute the analysis script (can be on a different node/job) pointing at the run directory. This script:

1. Reads `run_config.json`, the velocity field, and the snapshot manifest.
2. Streams through snapshots sequentially:

   * Loads one `.npz` snapshot at a time.
   * Immediately writes the corresponding theta+velocity overlay frame (PNG) to disk.
   * Dispatches the snapshot path to a worker pool that computes scalar structure functions, Yaglom statistics, and instantaneous dissipation. Workers use the same `--n-workers` value that also sets the FFT thread count to ensure consistent parallelism.
3. Computes the velocity-field structure functions (full range of `p` in `DEFAULT_ORDERS`) once per run and saves the results/plots to `analysis/velocity_structure_functions.*`.
4. After all snapshots are processed, stitches the already-created frames into an MP4 via `ffmpeg` and aggregates reduced diagnostics into a single `.npz` file (times, `r`, `S_p`, Yaglom curves, dissipation series, metadata).

Because snapshots are streamed, neither the simulation nor the analysis phase ever stores more than one scalar field in memory.

**Outputs per run (mirroring the old behavior but now produced by the analysis step):**

* `fields/`: velocity field(s), theta snapshots (`.npz`), snapshot manifest.
* `analysis/`: velocity quick-look, dissipation time series, scalar/Yaglom plots for each saved time, PNG frames, MP4 movie, and `reduced_diagnostics_*.npz` for later synthesis.

---

### 6. SLURM integration and parameter sweeps

1. Set up a SLURM batch script (or small launcher) that:

   * Requests one node per ((\alpha, \kappa)) pair (e.g., 16 nodes for 4 alphas × 4 kappas).
   * On each node, runs a **single** instance of `runSingleAlphaKappa.py` with the appropriate `alpha` and `kappa`.
2. Example parameter grid (hard-code or read from a config file):

   * `alpha` in ({1/6, 1/3, 1/2, 2/3})
   * `kappa` in some set of four values (to be defined; make it easy to edit).
3. Ensure that:

 * Each job writes to a unique output directory (e.g. by including `alpha` and `kappa` in the path).
 * No race conditions or filename collisions occur across nodes.

Once simulations finish, schedule post-processing jobs that run `analyzeAlphaKappaRun.py --run-dir <output>` on each directory. These can run on the same hardware but do not require the full walltime budget because they only read the saved fields.

---

### 7. Organization and style

* Follow existing module and directory structure.
* Reuse existing helper functions for:

  * Structure functions
  * Yaglom-like diagnostics (if any)
  * Anomalous dissipation
  * Plotting and movie making
* Add docstrings and comments that briefly explain:

  * What `runSingleAlphaKappa.py` does
  * What each major function returns
  * The meaning of each stored quantity in the `.npz` file

---

### 8. 4096² campaign blueprint

* **Parameter grid:** `alpha = [1, 2, 3, 4, 5]/6` and `kappa = 10**[-4, -3.5, -3, -2.5]`.
* **Run settings:** `grid=4096`, `t_end=20`, `n_save=200` (or `dt_save` if desired), `n_workers=128`, `cfl=0.6`, `integrator=etdrk4`, `dtype=float64`, `lam_min=2`, `lam_max=grid`, `velocity_urms=1.0`. The velocity generator therefore fills the entire spectral band from the largest to the smallest wavelengths.
* **Per-run logging:** Every call to `run_single_alpha_kappa.py` writes progress to `<run-dir>/run.log`. Pass `--quiet` (the batch script does this automatically) to keep stdout empty while still recording detailed progress in the log file.
* **Command generation:** `scripts/generate_alpha_kappa_campaign.py` builds sbatch commands and a manifest for the full grid. Edit the CLI defaults if the grid, worker count, or cadence needs to change; the script writes `campaigns/alpha_kappa_4096/commands.txt` and `manifest.json`.
* **Batch launcher:** `scripts/run_single_alpha_kappa.sbatch` now defaults to the 4096² production settings above (including `--quiet`). Submit a specific run with

  ```bash
  sbatch --export=ALPHA=0.3333333333,KAPPA=1e-3,GRID=4096,T_END=20,N_SAVE=200,N_WORKERS=128,CFL=0.6,INTEGRATOR=etdrk4,DTYPE=float64,LAM_MIN=2,LAM_MAX=4096,TAG=my_tag scripts/run_single_alpha_kappa.sbatch
  ```

  Use the generated `commands.txt` to launch the entire campaign (one command per run).
