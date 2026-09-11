# tripyview `workbench` branch — bug report

**Date:** 2026-07-10
**Branch audited:** `workbench` (commit at time of audit: `origin/workbench` HEAD)
**Method:** full read-through of all `tripyview/sub_*.py` modules (~28k lines), split across four parallel reviews (mesh/data core, transport/transect/MOC diagnostics, plotting, driver/utility/CLI), followed by manual verification of the four highest-severity claims against the actual source.
**Scope:** correctness bugs and clear inconsistencies only — not style, missing docs, or missing type hints.

This report does not fix anything; it documents findings and recommended fixes for someone to apply and test against real FESOM2 output before `workbench` is promoted to `main`.

---

## Severity legend

- 🔴 **Critical** — crashes or silently corrupts output under the function's own **default** arguments, on a documented/advertised feature.
- 🟠 **High** — crashes or silently corrupts output under common (non-default but not exotic) settings.
- 🟡 **Medium** — real bug, but on a narrow, rarely-exercised code path, or degrades gracefully.
- ⚪ **Low / cosmetic** — minor correctness issue or inconsistency, small practical impact.

Findings marked **[verified]** were confirmed by directly reading the code during this audit. Findings marked **[reported]** come from the sub-agent review and are stated with high confidence but were not independently re-read line-by-line.

---

## 🔴 Critical

### 1. `sub_dmoc.py:442-446` — `NameError`, density-space MOC crashes under default arguments **[verified]** — ✅ FIXED

```python
# line ~415-416
data_dMOC = xr.merge([data_dMOC, data_div], combine_attrs=which_combineattrs)
del(data_div)          # <-- data_div deleted here
gc.collect()

# line ~420-459, inside `if (do_bolus): ... if not do_wdiap:`
e_i = xr.DataArray(mesh.e_i[:,0], dims=['elem'])
aux_dmoc_div =                data_div['dmoc_bolus'].isel(nod2=e_i)   # <-- data_div no longer exists
e_i = xr.DataArray(mesh.e_i[:,1], dims=['elem'])
aux_dmoc_div = aux_dmoc_div + data_div['dmoc_bolus'].isel(nod2=e_i)   # same
e_i = xr.DataArray(mesh.e_i[:,2], dims=['elem'])
aux_dmoc_div = aux_dmoc_div + data_div['dmoc_bolus'].isel(nod2=e_i)   # same
```

`data_div` is deleted before this block runs. The dataset that actually holds `'dmoc_bolus'` at this point is `data_div_bolus` (loaded and renamed a few lines above, at line ~422-423). `load_dmoc_data()`'s own defaults are `do_bolus=True`, `do_wdiap=False`, so this crashes on essentially any call requesting `which_transf='dmoc'` or `'inner'` — the density-space MOC diagnostic credited to Dmitry Sidorenko in the README.

**Fix:** replace `data_div` with `data_div_bolus` on all three lines:

```python
e_i = xr.DataArray(mesh.e_i[:,0], dims=['elem'])
aux_dmoc_div =                data_div_bolus['dmoc_bolus'].isel(nod2=e_i)
e_i = xr.DataArray(mesh.e_i[:,1], dims=['elem'])
aux_dmoc_div = aux_dmoc_div + data_div_bolus['dmoc_bolus'].isel(nod2=e_i)
e_i = xr.DataArray(mesh.e_i[:,2], dims=['elem'])
aux_dmoc_div = aux_dmoc_div + data_div_bolus['dmoc_bolus'].isel(nod2=e_i)
```

**Applied:** as above, plus a fourth stray reference at (now) line 453 (`aux_dmoc_div.assign_attrs(data_div['dmoc_bolus'].attrs)`) that wasn't caught in the initial audit — also corrected to `data_div_bolus`.

---

### 2. `sub_tripyrun.py:293` — `KeyError`, `-d`/`-v` CLI flags crash on first use **[verified]** — ✅ FIXED

```python
for vname in inargs.variable:
    cnt, cnt_max = -1, -1
    for values in webpages["analyses"][analysis_name].values():   # <-- KeyError if analysis_name not present yet
        cnt_max = np.maximum(cnt_max, values['cnt'])
        if values['variable']==vname:
            cnt = values['cnt']
            break
    if cnt==-1:
        print(" --> could not find variable: {vname} in loaded webpage. This variable will be attached if it exist")
        cnt=cnt_max+1
```

The code assumes `webpages["analyses"][analysis_name]` already exists. On a fresh run (no prior `.json`, so `webpages["analyses"] = {}`), or when `-v` targets a diagnostic not yet in an existing results file, this raises `KeyError` — before the fallback logic that's clearly designed to handle exactly this case (`if cnt==-1: ... "will be attached if it exist"`) ever gets to run.

**Fix:** use `.get()` with a default empty dict so the loop simply doesn't execute and falls through to the existing fallback:

```python
for values in webpages["analyses"].get(analysis_name, {}).values():
```

**Applied:** as above (both the scan-loop read and the driver-call read at the old line 301 now use `.get(analysis_name, ...)`), plus: (1) an explicit message printed when `analysis_name` has no prior entry at all, naming the driver, the JSON file, and the variable(s) about to be created fresh; (2) fixed a related bug on the same line — the "could not find variable" message was missing its `f` prefix, so `{vname}` was never substituted.

**Scope note (confirmed via testing):** this bug only triggers when `-d` and `-v` are used *together* targeting a driver with no prior JSON entry. A plain `tripyrun file.yml` (no flags) never reaches this code path — it takes the separate "run everything" branch — so normal full-run usage was never affected.

---

### 3. `sub_transp.py:1325` — `NameError`, zonal heat-flux crashes under default arguments **[verified]** — ✅ FIXED

```python
for ix, lon_i in enumerate(zhflx.lon):
    if do_info: print('{:+06.1f}|'.format(lat_i), end='')   # <-- lat_i is undefined here
```

Inside `calc_zhflx_box_fast_lessmem`, the loop variable is `lon_i`; `lat_i` is a copy-paste leftover from the sibling function `calc_mhflx_box_fast_lessmem` (meridional flux) and is never defined in this function's scope. This function's own defaults are `do_info=True`, `do_parallel=False`, so it crashes on the very first iteration of any non-parallel zonal heat-flux computation (`zhflx` driver).

**Fix:**

```python
if do_info: print('{:+06.1f}|'.format(lon_i), end='')
```

**Applied:** as above. Confirmed the identical print string also appears at `sub_transp.py:438` and `:676`, inside the two meridional (`calc_mhflx_box_fast`/`calc_mhflx_box_fast_lessmem`) sibling functions, where `lat_i` is genuinely the loop variable — those two are correct and were left unchanged.

---

### 4. `sub_mesh.py:1280-1283` — vertex area weights collapse to ~0 on FESOM1.4 fallback path **[verified]** — ✅ FIXED

```python
self.n_area = np.zeros((self.n2dn))
count_e = 0
for idx in self.e_i.flat:
    self.n_area[idx] = self.n_area[idx] + e_area_x3[count_e]
    count_e = count_e+1
    self.n_area = self.n_area/3.0     # <-- indented inside the loop, runs 3*n2de times
del e_area_x3, count_e
```

In `compute_n_area()`'s fallback branch (taken when no `griddes.nc` is found, i.e. the FESOM1.4/CMIP6-conversion path), the `/3.0` normalization is indented one level too far and executes on every loop iteration (`3 * n2de` times) instead of once after the loop. `n_area` ends up divided by `3**(3*n2de)`, i.e. numerically zero for any realistic mesh size. Any area-weighted computation on such a mesh (e.g. binned AMOC diagnostics on cmorized FESOM1.4 output) will silently produce garbage.

**Fix:** dedent the normalization line to run once, after the loop:

```python
self.n_area = np.zeros((self.n2dn))
count_e = 0
for idx in self.e_i.flat:
    self.n_area[idx] = self.n_area[idx] + e_area_x3[count_e]
    count_e = count_e+1
self.n_area = self.n_area/3.0
del e_area_x3, count_e
```

---

### 5. `sub_climatology.py:117` — climatology comparisons silently use lat where lon is expected **[reported, high confidence]** — ✅ FIXED

```python
data_lon = data[coord_lon].expand_dims(...)              # line 116, correct
data_lon = data_lat.transpose(coord_zlev, coord_lat, coord_lon)   # line 117, overwrites with data_lat!
```

`data_lon` is computed correctly, then immediately clobbered by a transpose of `data_lat` instead of `data_lon`. Downstream, `gsw.SA_from_SP(..., data_lon, data_lat)` (line ~122) and the potential-density/sigma branch (line ~143) both receive latitude values in the longitude argument slot. This silently corrupts every `load_climatology()` call for `temp`/`sst`/`pdens`/`sigma*` — the PHC3 climatology-comparison diagnostics.

A related, likely-connected issue at the same call site: line ~143 passes the raw `data_depth` where line ~122's (correct) call passes the derived sea-pressure `data_p` — worth checking both together.

**Fix:**

```python
data_lon = data_lon.transpose(coord_zlev, coord_lat, coord_lon)
```
and verify the line-143 call site uses `data_p` consistently with line 122.

**Applied:** as above, plus the related line-143 fix: `gsw.SA_from_SP(data[vname_salt].data, data_depth, data_lon, data_lat)` → `..., data_p, ...`. Confirmed against the `gsw.SA_from_SP(SP, p, lon, lat)` signature — its second argument must be sea pressure in dbar (as correctly computed via `gsw.p_from_z()` and used at line 122), not raw depth in meters; the `pdens`/`sigma*` branch was passing depth directly, understating/distorting Absolute Salinity (and everything downstream: `CT_from_pt`, `gsw.rho`) with growing error at depth.

---

## 🟠 High

### 6. `sub_utility.py:290` / `sub_dmoc.py:634` — `TypeError`, stale keyword argument **[reported]** — ✅ FIXED

`calc_basindomain_fast()`'s current signature takes `do_exclude`/`exclude_list`, not `exclude_meditoce`. `sub_dmoc.py:634` still calls it with `exclude_meditoce=exclude_meditoce`. A second, correct call site exists at `sub_dmoc.py:1012` using the current signature — confirming line 634 is simply stale and was missed when the signature changed.

**Fix:** update the call at `sub_dmoc.py:634` to match the current `calc_basindomain_fast()` signature, mirroring the call at line 1012.

**Applied:** rather than translating only the call site, `calc_dmoc()`'s own public signature was changed from `exclude_meditoce=False` to `do_exclude=False, exclude_list=['ocean_basins/Mediterranean_Basin.shp', [26,42,39.5,47]]` — matching `calc_dmoc_dask()`'s signature exactly, so both sibling functions now share one interface. Confirmed via repo-wide grep that no active caller (`.py`/`.ipynb`/`.yml`) passes `exclude_meditoce=` to `calc_dmoc()` — the only other reference was a commented-out notebook example — so this is a safe interface change. Both functions' docstrings (lines ~559, ~954) were updated to document `do_exclude`/`exclude_list` instead of the stale `exclude_meditoce` entry.

**Known pre-existing rough edge (not fixed):** `sub_utility.py:313`, inside `calc_basindomain_fast()`'s `isinstance(which_moc, shp.Reader)` branch, still assigns a local `exclude_meditoce=False` that is never read anywhere in the function — dead/vestigial code from before the `do_exclude`/`exclude_list` refactor. Harmless but confusing; left for the cosmetic cleanup pass.

---

### 7. `sub_transect.py:1366` — `NameError`, typo `mesh_zmid` **[reported]** — ✅ FIXED

```python
... mesh.zmid[:-1] - mesh_zmid[1:] ...   # missing dot
```

Undefined name; every other reference in the codebase uses `mesh.zmid`. Triggers whenever `calc_transect_Xtransp` runs with velocity data on full levels (`nz`) rather than mid-levels (`nz1`).

**Fix:** `mesh_zmid` → `mesh.zmid`.

**Applied:** as above.

---

### 8. `sub_zmoc.py:414` and `:775` — `KeyError` in info-summary block **[reported]** — ✅ FIXED

```python
zmoc['zmoc'].isel(...)['moc'].min()   # should be ['zmoc'], not ['moc']
```

Present identically in both `calc_zmoc()` (line 414) and `calc_zmoc_dask()` (line 775). Triggers when `do_info=True` (the default) and `which_moc` is `'pmoc'`/`'ipmoc'` with no time dimension: the MOC computation itself succeeds, but the function crashes on the summary print right before returning.

**Fix:** `['moc']` → `['zmoc']` in both locations.

**Applied:** `zmoc['zmoc'].isel(...)['moc']` (a DataArray indexed a second time by string, which is invalid) → `zmoc.isel(...)['zmoc']` in both locations, mirroring the working `amoc`/`aamoc`/`gmoc` pattern two lines above (`.isel()` on the Dataset, then select the `'zmoc'` variable).

---

### 9. `sub_tripyrundriver.py:477` — wrong notebook template for `hquiver` **[reported]** — ✅ FIXED

```python
loop_over_param(..., exec_template='hslice')   # should be 'hquiver'
```

`drive_hquiver()`'s branch for when no `depths`/`depth` key is given in the YAML still points at `'hslice'` — the template it was cloned from — while the other three branches in the same function correctly use `'hquiver'`.

**Fix:** `exec_template='hslice'` → `exec_template='hquiver'` in that branch.

**Applied:** as above. Note `sub_tripyrundriver.py`'s module-level `exec_papermill(..., exec_template='hslice')` default (line 22) and `drive_hslice()`'s own four call sites (lines 275-291) are unrelated/correct and were left untouched.

---

### 10. `sub_tripyrundriver.py:1106` — wrong notebook template for `transect_mmean_clim` **[reported]** — ✅ FIXED

```python
exec_template='transect_zmean_clim'   # hardcoded; should track analysis_name
```

`drive_transect_zm_mean_clim()` is registered for both `transect_zmean_clim` and `transect_mmean_clim` (see the dispatch table in `sub_tripyrun.py`), but hardcodes the zmean template regardless of which one was requested. Its sibling `drive_transect_zm_mean()` correctly uses `exec_template=analysis_name`. Running `-d transect_mmean_clim` executes `template_transect_zmean_clim.ipynb` instead of the separately-existing `template_transect_mmean_clim.ipynb`.

**Fix:** `exec_template='transect_zmean_clim'` → `exec_template=analysis_name`.

**Applied:** as above. Confirmed both `template_transect_zmean_clim.ipynb` and `template_transect_mmean_clim.ipynb` exist in `templates_notebooks/`.

---

### 11. `sub_plot.py:6955` — always-true guard corrupts log-scale colorbar auto-ranging **[reported]** — ✅ FIXED

```python
# if not do_rescale=='log10' and not do_rescale=='slog10':   <- commented-out original intent
if not isinstance(do_rescale,str) or not isinstance(do_rescale, np.ndarray):
```

A value can never be both a `str` and an `np.ndarray` at once, so `not A or not B` is unconditionally `True` regardless of `do_rescale`. This is a De Morgan inversion of the commented-out original condition. The linear-only cmin/cmax decimal-rounding block that follows now always runs, including for `do_rescale='log10'`/`'slog10'` and custom `np.ndarray` rescale steps, silently shifting auto-computed log-scale colorbar bounds away from the true data extrema whenever no explicit `crange` is passed.

**Fix:** restore the intended condition, e.g.:

```python
if do_rescale not in ('log10', 'slog10'):
```

**Applied differently than suggested above — the suggested one-liner itself crashes.** `do_rescale` can legitimately be a multi-element `np.ndarray` (custom rescale bin edges, see `mcolors.BoundaryNorm(do_rescale, ...)` at line ~4737). Both `do_rescale not in ('log10','slog10')` (the fix suggested above) and the original commented-out intended condition (`not do_rescale=='log10' and not do_rescale=='slog10'`) raise `ValueError: truth value of an array... is ambiguous` when `do_rescale` is such an ndarray, since comparing an array to a string produces an elementwise array that Python's `in`/`not`/boolean context can't collapse to a single bool. Verified this with a standalone repro before applying. Fixed with an isinstance-guarded condition that short-circuits before ever comparing an ndarray to a string, and — consistent with this finding's own description that the block should be linear-only — also skips the decimal-rounding block for ndarray custom rescale steps, not just the two log strings:
```python
if not (isinstance(do_rescale, str) and do_rescale in ('log10', 'slog10')) and not isinstance(do_rescale, np.ndarray):
```

---

### 12. `sub_plot.py:5649-5652` — mutates caller's dict, breaks multipanel streamline plots **[reported]** — ✅ FIXED

```python
if 'lw_min' in streaml_opt:
    lw_min = streaml_opt['lw_min']
    del(streaml_opt['lw_min'])   # mutates the caller's dict object
```

`streaml_opt` is the same dict object passed once into `plot_hslice(...)` and reused, unmodified by the caller, across the per-panel loop in a multipanel figure. On the first panel, `lw_min`/`lw_max` are deleted from the caller's dict; every subsequent panel's lookup silently fails and falls back to hardcoded defaults (0.25/5.0) with no warning.

**Fix:** operate on a local copy instead of the caller's dict, e.g. at the top of the function:

```python
streaml_opt = dict(streaml_opt) if streaml_opt else {}
```

**Applied:** as above, at the top of `do_plt_streaml_reg()` right after `h0=None`.

---

### 13. `sub_data.py:979-984` — wrong coordinate/length check for full-level (`nz`) area weights **[reported]** — ✅ FIXED

The `'nz1'` (mid-level) branch in `do_gridinfo_and_weights()` correctly checks against `len(mesh.zmid)` and, in the subset case, reads the index coordinate `data['nzi']`. The `'nz'` (full-level) sibling branch instead checks `data.sizes['nz'] == len(mesh.zmid)` (should compare to `len(mesh.zlev)`, since `mesh.n_area` has `len(mesh.zlev)` rows) and its `else` reads `data['nz']` — actual depth-in-meters values, possibly already dropped by an earlier `drop_vars('nz')` — instead of `data['nzi']`.

**Failure:** loading any full-level variable (e.g. `w`, `Kv`) with `do_hweight=True` (default) either raises `KeyError`, or uses depth-in-meters cast to `uint8` (wrapping mod 256) as row indices into `mesh.n_area`, producing wrong horizontal area weights.

**Fix:** mirror the `nz1` branch's logic — compare against `len(mesh.zlev)`, and use `data['nzi']` in the subset `else` case.

**Applied:** as above.

---

## 🟡 Medium

### 14. `sub_dmoc.py:1284` — off-by-one in bottom-topography mask (dask path only) **[reported]** — ✅ FIXED

```python
mesh.zlev[mesh.e_iz-1]   # extra, unwarranted -1
```

`mesh.e_iz` is already 0-based and used directly (no `-1`) everywhere else in the codebase, including the non-dask `calc_dmoc()` sibling in the same file. This extra `-1` in `calc_dmoc_dask()` makes the `do_botmax_z` bottom-topography mask one vertical level too shallow, giving results that differ depending on whether `do_parallel` is used.

**Fix:** `mesh.zlev[mesh.e_iz-1]` → `mesh.zlev[mesh.e_iz]`.

**Applied:** as above. Confirmed via `grep` that `calc_dmoc()`'s own uses of `mesh.e_iz` (lines 861, 872) never subtract 1, consistent with the fix.

---

### 15. `sub_transp.py:1392` — off-by-one drops northernmost latitude bin **[reported]** — ✅ FIXED

```python
for bini in range(lat_i.min(), lat_i.max()):   # excludes lat_i.max()
```

Python's `range()` excludes its upper bound, so the bin at `lat_i.max()` is never summed and stays at its zero-initialized value; that zero then propagates through the subsequent cumulative sum in `calc_gmhflx`, understating global meridional heat flux near the domain edge. (The apparently-preferred sibling `calc_gmhflx_box` bins differently and does not have this bug.)

**Fix:** `range(lat_i.min(), lat_i.max()+1)`.

**Applied:** as above.

---

### 16. `sub_plot.py:6598` — typo produces blank colorbar label **[reported]** — ✅ FIXED

```python
elif 'short_name' in loc_attrs:
    c_label = cb_label+loc_attrs['short_name'].capitalize()   # should be cb_label
```

Writes to `c_label`, a variable that's never read; `cb_label` stays empty. Any variable whose attrs carry `short_name` but no `long_name` (common for FESOM diagnostics) gets a colorbar with a blank/missing name label when no explicit `cb_label` is passed.

**Fix:** `c_label` → `cb_label`.

**Applied:** as above.

---

### 17. `sub_plot.py:6365` — user's custom density-axis ticks silently ignored **[reported]** — ✅ FIXED

```python
elif ii in ['ysig_majorticks']:
    yexp_majorticks = grid_optdefault[ii]   # should be ysig_majorticks
```

Copy-paste bug in `do_plt_gridlines`: a user-supplied `grid_opt={'ysig_majorticks': [...]}` for a density-coordinate (`dmoc`) plot updates the unused `yexp_majorticks` variable instead of `ysig_majorticks`, which is what's actually read later for the density-axis tick labels. The override is silently ignored.

**Fix:** `yexp_majorticks` → `ysig_majorticks`.

**Applied:** as above. Confirmed `ysig_majorticks` is genuinely read downstream (`hax_ii.set_yticks(ysig_majorticks, ...)` and two further uses at lines ~6417-6423) — this was a live bug, not dead code.

---

### 18. `sub_mesh.py:3568` — wrong index in land-sea-mask contour tracer **[reported]** — ✅ FIXED

```python
if canreturn[0]==0 and isreturn[cur]==0:   # should be canreturn[cur]
```

In the numba land-sea-mask boundary tracer, this checks the fixed flag for node index 0 instead of the current node (`cur`) being traced — inconsistent with the surrounding `canreturn[cur]=...` assignments. At boundary nodes with more than 2 neighbors (a documented, if rare, topology), traversal decisions incorrectly depend on node 0's flag state rather than the current node's.

**Fix:** `canreturn[0]` → `canreturn[cur]`.

**Applied:** as above, after tracing the full `njit_lsmask_trace_loops()` function to confirm: `canreturn` persists across the whole trace (all `start` values), so once any node sets `canreturn[0]`, the junction-vs-normal branch decision is silently wrong for every other node in the mesh, not just the one containing node 0. Only manifests on meshes with a genuine boundary pinch point (a node with 4 valid neighbor slots).

---

### 19. `sub_data.py:639` — dead branch, out-of-range depths not clamped for full-level variables **[reported]** — ✅ FIXED

```python
if   dim_vert == 'nz1': auxdepth = np.clip(auxdepth, abs(mesh.zmid[0]), abs(mesh.zmid[-1]))
elif dim_vert == 'nz ': auxdepth = np.clip(auxdepth, abs(mesh.zlev[0]), abs(mesh.zlev[-1]))   # trailing space typo
```

`dim_vert` is always assigned exactly `'nz'` (no trailing space) elsewhere, so this branch never fires. For full-level variables, requested interpolation depths outside the mesh's valid z-range are never clamped, so `data.interp(...)` silently returns `NaN` instead of clamping to the nearest valid level (as the `nz1` branch correctly does).

**Fix:** `'nz '` → `'nz'`.

**Applied:** as above. Confirmed `do_gridinfo_and_weights()` (`sub_data.py:922`) always assigns exactly `dimn_v = 'nz'`, which flows through as `dim_vert` at the call site.

---

## ⚪ Low / cosmetic

### 20. `sub_index.py:103-104,152` — dead branch from bitwise-not misuse **[reported]** — ✅ FIXED

`do_elem` is set to `False` on both branches that assign it, then checked via `if ~do_elem:`. In Python, `~False == -1`, which is truthy — so this condition is always `True` regardless of `do_elem`'s actual value, permanently disabling the intended toggle. Only affects the optional `do_checkbasin=True` debug plot's triangle-centroid variant (not production output).

**Fix:** use proper boolean logic, e.g. `if not do_elem:`.

**Applied:** as above. Verified `bool(~False)` and `bool(~True)` are both `True` in Python, confirming the guard was unconditionally true regardless of `do_elem`.

### 21. `sub_warmup_numba.py:224-228` — unguarded numba warm-up calls at import time **[reported]** — ✅ FIXED

`warmup_grid_kernels()`, `warmup_vec_r2g_kernels()`, `warmup_lsmask()`, and `warmup_smoothing_kernels()` run unconditionally at package-import time without the try/except guard that `warmup_compute_x_nghbr_x()` uses. Any numba typing/signature drift in these specific kernels would crash `import tripyview` for every user instead of degrading to a warning.

**Fix:** wrap these four calls in the same try/except pattern already used for `warmup_compute_x_nghbr_x()`.

**Applied:** wrapped each of the four module-level calls in its own try/except with a warning print (one exception handler per function call, coarser-grained than `warmup_compute_x_nghbr_x()`'s per-kernel wrapping, but consistent with the "don't let a single kernel's compile failure crash import" intent). Verified `import tripyview` still prints all five "Warming up..." lines with no warnings.

### 22. `sub_utility.py:1354` — off-by-one in interactive box navigation **[reported]** — ✅ FIXED

```python
self.idx_box = np.min([self.idx_box+1, len(self.box_list)])   # should cap at len(self.box_list)-1
```

Lets `idx_box` reach `len(box_list)`, causing an `IndexError` on the next `box_list[idx_box]` access in the interactive polygon-selection GUI helper `select_scatterpts_depth`. GUI-only, no effect on batch/`tripyrun` diagnostics.

**Fix:** `len(self.box_list)` → `len(self.box_list)-1`.

**Applied:** as above.

### 23. `sub_plot.py:7172` — sign error in histogram bin-midpoint formula **[reported]** — ✅ FIXED

```python
bin_m = bin_e[:-1]+(bin_e[:-1]-bin_e[1:])/2   # computes left_edge - width/2 instead of + width/2
```

Shifts every histogram-based auto cmin/cmax (default `chist=True`, outlier-robust auto color range) one full bin width toward smaller values. With the default `cbin=1000`, this is roughly a 0.1% offset — mathematically wrong but unlikely to be visually noticeable.

**Fix:** `(bin_e[:-1]-bin_e[1:])/2` → `(bin_e[1:]-bin_e[:-1])/2`.

**Applied:** as above.

### 24. Minor items not independently verified, worth a quick look

- **`sub_tripyrundriver.py:97`** — ✅ FIXED. Confirmed real: the established convention in this function (verified at lines 41, 45, 51) is `str_proj1` = filename-safe form (feeds `str_all1`), `str_proj2` = human-readable form (feeds `str_all2`). Line 97, in the `dmoc_wdiap`/`dmoc_srfcbflx` branch, used `str_proj1` in both. Fixed: `str_all2 = f"{str_isop2}{str_proj1}{str_mon2}"` → `f"{str_isop2}{str_proj2}{str_mon2}"`. Cosmetic-only (webpage display title), no effect on filenames or figure content.
- **`sub_tripyrun.py:52-56`** — ✅ FIXED. `render_experiment_html()` now renders the template into `output` *before* opening the file, then writes it inside a `with open(...) as ofile:` block. This is a stronger fix than just adding a `with`: opening in `"w"` mode truncates the file immediately, so if `template.render()` raised (e.g. malformed `webpages` dict), the previous run's `.html` report was being wiped to 0 bytes before any new content existed. Rendering first means a failed render now leaves the last successful report on disk untouched instead of destroying it — relevant here since `tripyrun` calls this after every driver specifically to support resuming/re-rendering a partially-finished run.
- **`sub_tripyrun.py` CLI help text** (around line 98) — ✅ FIXED. `' - ghflx, mhflx, zhflx \n'` → `' - gmhflx, mhflx, gzhflx \n'`, matching the actual `analyses_driver_list` keys (`sub_tripyrun.py:235-236`).
- **`sub_data.py:2427`** — ✅ FIXED, confidence upgraded from low to confirmed. `dx, dy = Rearth*dlon*rad*np.cos((lat)/2.0*rad), ...` halved latitude before taking cosine, with no physical basis for a zonal-cell-width scaling factor. Confirmed by finding the same computation done correctly elsewhere in the codebase (`sub_mesh.py:1104-1107`: `cos(lat*rad)`, no halving). Fixed: `np.cos((lat)/2.0*rad)` → `np.cos(lat*rad)`. As previously noted, only affects the `w_A` metadata coordinate written onto coarse-grained output (mis-stating each output cell's area in m²) — confirmed the actual coarse-graining/binning math uses the original mesh's native `data['w_A']`, not this recomputed `dA`.
- **`sub_transect.py`, `calc_transect_zm_mean_chnk`** — investigated further, left **unfixed**. Confirmed the shape mismatch is real *if* the 3D branch is reached: `chnk_d[:,:,nod_i]` has shape `(ntime, nlev)` while `binned_d[0,:,:,jj]` has shape `(nlev, ntime)` — these should raise a broadcast `ValueError` unless `ntime==nlev` coincidentally, in which case they'd silently combine transposed data. However, whether this branch is actually reachable depends on the true axis order dask hands `chnk_d` at runtime through the caller (`calc_transect_zm_mean_dask`, lines 2508-2529), which prepares inputs in a way that's hard to pin down statically: `chnk_lonlat`/`chnk_ispbnd` get one synthetic `[None,:]` axis to "match dimensionality," `drop_axis=[0]`'s inline comment ("drop dim nz1") appears to contradict the text comment two lines above it ("drop_axis = [1]"), and there's no explicit `.transpose()` anywhere to pin down the order. Resolving this needs a runtime test against real multi-timestep FESOM2 data (`do_parallel=True`, transect zonal-mean on time-dependent output), not further static reading.

---

## Recommended order of work

Given the goal of promoting `workbench` to `main`, address in this order:

1. Findings **1-5** (🔴 Critical) — these break advertised, default-path behavior of `dmoc`, `zhflx`, the `tripyrun -d/-v` CLI, and climatology comparisons. Fix and smoke-test each before anything else.
2. Findings **6-13** (🟠 High) — real crashes/corruption on common but non-default settings; each is a small, isolated fix.
3. Findings **14-19** (🟡 Medium) — fix opportunistically; narrower blast radius.
4. Findings **20-24** (⚪ Low/cosmetic) — low priority, batch these into a cleanup pass.

None of these require design changes — every fix above is a one-to-few-line correction. Suggest adding a regression test (or at minimum a manual `tripyrun` smoke-test run covering `dmoc`, `zhflx`, `hquiver`, `transect_mmean_clim`, and a fresh `-d/-v` invocation) once fixed, since the underlying reason these survived is that none of these specific code paths (non-default arguments, first-run CLI states, particular basin/mesh combinations) are covered by any existing test.

---

# Second pass — 2026-09-10 (usability / performance scan)

Found during a repo-wide usability/performance scan. Each finding below was reproduced by running it (not just by reading) and re-tested after the fix on the `core2_pool` mesh / the `dvd_core2_tke+idemix_GM1000_01` run.

### 25. `setup.py` — non-editable installs ship no templates, shapefiles or backgrounds 🟠 — ✅ FIXED

`packages=['tripyview']` with no `package_data` meant `pip install .` (what the `Dockerfile` runs) installed only the `.py` files: no `templates_notebooks/`, `templates_html/` (both live at the repo root), `tripyview/shapefiles/` or `tripyview/backgrounds/`. `tripyrun` could not find its templates and every region-mask diagnostic would fail. Only `pip install -e .` worked.

**Applied:** `setup.py` maps the two template directories into the package (`tripyview.templates_notebooks`, `tripyview.templates_html`) and adds `package_data` for shapefile components one level deep (`shapefiles/<category>/<name>.*`, which keeps ~150 MB of untracked local data such as `shapefiles/marineregions.org/` out of builds) and `backgrounds/*`. Template paths are now resolved once in `sub_tripyrundriver.py`: in-package when installed, repo root for an editable checkout (unchanged behaviour); the duplicate definition in `sub_tripyrun.py` was removed. The default `Results/` folder stays in the repo for a dev checkout and becomes the current working directory for an installed package (instead of writing into `site-packages`). Verified with a non-editable install into a scratch target and with the existing `egg-link` checkout.

### 26. `sub_utility.py` `do_boxmask` / `do_boxmask_dask` — `ndarray` polygon boxes crash 🟡 — ✅ FIXED

`if box == None or box == 'global'` is elementwise for an `np.ndarray` box, so the documented `[2 x npts]` polygon form (e.g. `[np.array(...), 'name']` in a `box_list`) raised `ValueError: The truth value of an array ... is ambiguous`. **Applied:** `box is None or (isinstance(box, str) and box == 'global')` in both functions; same type-safe check for the equivalent condition in `sub_transect.py` (`load_zmeantransect_fesom2`, `calc_transect_zm_mean_dask`).

### 27. `sub_utility.py` `do_boxmask` — `MultiPolygon` boxes crash on Shapely ≥ 2 🟡 — ✅ FIXED

`for p in box:` — a `MultiPolygon` is no longer iterable in Shapely 2 (installed: 2.0.7), and the mask was allocated with `mesh.n2dn` even for `do_elem=True`. **Applied:** `for p in box.geoms`, mask sized `mesh_x.size`. Same iteration fix in `sub_index.py` `plot_index_region`.

### 28. Multi-part shapefiles turned into a single garbage polygon 🟡 — ✅ FIXED

`Polygon(shape.points)` concatenates all parts (islands, disjoint pieces) of a shapefile shape into one self-intersecting ring and ignores holes. Affected 3 of 55 tracked shapefiles: `iho_def/Greenland_Sea` (1174 parts; old mask 883 nodes vs. correct 2024), `iho_def/Iceland_Sea` (769 vs 830), `iho_def/Norwegian_Sea` (2239 vs 2250). **Applied:** new helper `shp_shape_to_geom()` (`sub_utility.py`) builds the shapely geometry from the shape's `__geo_interface__`; used in `do_boxmask`, `do_boxmask_dask` and `plot_index_region`. The 52 single-part shapefiles give bit-identical masks to before; the 3 multi-part ones now match an independent geopandas reference exactly.

### 29. `sub_data.py` `compute_optimal_chunks` — invalid default and unformatted error ⚪ — ✅ FIXED

Default `opti_dim='hori'` is not a value the function accepts (it would raise when called with defaults and a client), the docstring listed options the code does not handle, and the error used `r'...'` instead of `f'...'` so `{opti_dim}` was never substituted. All existing callers pass `'h'`/`'v'`, so this was latent. **Applied:** default `'h'`, docstring lists the real options, f-string error naming the accepted values.

## Robustness / usability fixes (same pass, 2026-09-10)

### 30. `tripyrun` hid failed notebooks and exited 0 🟠 — ✅ FIXED

`exec_papermill` (`sub_tripyrundriver.py`) caught every exception, printed one line, then still registered the (missing) figure in the html report, and `tripyrun` exited 0, so a SLURM job with failed diagnostics looked successful; on success it printed a misleading `Data found`. **Applied:** failures print the notebook path plus `ename: evalue` (full traceback stays in the executed notebook), are collected in `failed_notebooks`, and a failed notebook without a figure is no longer linked in the html. `tripyrun()` lists the failed notebooks at the end and returns 1, which the console script turns into exit status 1. Successful runs are registered exactly as before (5 templates do not write the standard `save_fname`, so file existence is only checked for failed runs). Verified with a real `tripyrun` run whose `hmesh` notebook fails.

### 31. `tripyrun` resume JSON could be truncated by a killed job 🟡 — ✅ FIXED

The `<tripyrun_name>.json` was written with `open(..., "w")`; a walltime kill during the write left a truncated file and the next resume crashed in `json.load`. **Applied:** write to `<file>.tmp`, then `os.replace` (atomic).

### 32. Misspelled options were silently ignored 🟡 — ✅ FIXED

`do_axes_arrange(**kwargs)` (receives every `ax_opt` dict) and `load_dmoc_data`, `calc_dmoc`, `calc_dmoc_dask` accepted `**kwargs` and never used them, so e.g. `ax_opt={'cb_poss': ...}` or `do_bolous=False` had no effect and no error. Real example found: old executed dmoc notebooks under `Results/` passed `exclude_meditoce=...` to `calc_dmoc`, which was dropped. **Applied:** a `UserWarning` naming the ignored option(s). Checked beforehand that no internal call, tracked template or local notebook passes an unknown key, so correct code stays silent.

### 33. Debug output on every import / load / plot ⚪ — ✅ FIXED

Removed leftover debug prints: repo path on every `import tripyview` (`sub_tripyrun.py`), `sel_levidx` and `ndimax=` on every depth-selecting `load_data_fesom2` call, `print(cinfo)` / `cmin, cmax =` / slog10 threshold / `print(plt_optdefault)` / `print(auxidx)` / `print(proj, box)` / log10 decimal limits in `sub_plot.py`, a shape print in `sub_3dsphere.py`. Gated on the existing `do_info`: `print(dmoc)` in `calc_dmoc`, lon/lat range in `calc_transect_zm_mean_dask`, and the eight `mesh_fesom2.compute_*` messages (now silent with `load_mesh_fesom2(..., do_info=False)`, also for cached meshes). The five numba warm-up lines on every import are gone; one line is printed only when the kernels actually had to be compiled (> 5 s). Kept on purpose: chunk progress counters, genuine warnings, `grid_interp_e2n` timing and the `vec_r2g_dask` message (deliberate info output without a `do_info` parameter).

### 34. Smaller items ⚪ — ✅ FIXED

- papermill kernel can be chosen with `TRIPYVIEW_KERNEL=<name>` (default stays `python3`); documented in the README together with the new exit-code behaviour.
- `load_mesh_fesom2` docstring had the `do_pickle`/`do_joblib` defaults swapped; `chnksize` docstrings said `1e6` instead of `6.5e6` (`plot_hslice`) / `3e6` (`plot_hmesh`, `plot_hquiver`).
- Removed the empty cookiecutter stub `tripyview/tripyview.py` (unreferenced).
- `sub_utility.contains` now uses `shapely.contains_xy` (identical to the deprecated `shapely.vectorized.contains`, which it falls back to on shapely < 2); masks verified identical for all 55 tracked shapefiles.

**Deliberately not changed:** `xr.set_options(keep_attrs=True)` at package import (many functions rely on attributes surviving arithmetic, e.g. labels and file names), and adding `__all__` to the modules (the `sub_*` modules pick up `np`, `xr`, `plt`, ... from each other via `import *`). Both would need a dedicated refactor with tests.

## Performance fixes — 2026-09-11

### 35. `compute_n_area` FESOM1.4 fallback: Python scatter-add loop → `np.bincount` — ✅ FIXED

`sub_mesh.py` accumulated 1/3 of every triangle area onto its 3 vertices with a Python loop over `3 x n2de` `(element, corner)` pairs. Measured on core2 (244659 elements, 733977 pairs, ~5.8 triangles per vertex): loop 313 ms, `np.add.at` 62 ms, `np.bincount(idx, weights=..., minlength=n2dn)` 2.2 ms. The obvious `n_area[e_i.ravel()] += ...` is **not** an option: fancy-index assignment is buffered, so only the last of the ~6 triangles sharing a vertex survives — measured 16.2% of the correct total area, silently. **Applied:** `np.bincount`, verified bit-identical to the old loop (max abs diff 0.0) through the real fallback branch (`do_f14cmip6=True`, no `griddes.nc`): 372 ms -> 5.4 ms, i.e. 69x; total area 3.643543e+14 m^2 either way.

### 36. Chunked plotting redrew the whole figure per chunk on non-interactive backends — ✅ FIXED

`do_plt_data`, `do_plt_bot`, `do_plt_mesh` and the two quiver/streamline paths called `hfig.canvas.draw_idle()` + `flush_events()` after every plotted chunk. On a GUI backend that shows the figure filling up; on `agg` / jupyter `inline` (papermill, tripyrun) `draw_idle()` is a full synchronous redraw of the complete figure, so the cost grows with the chunk count. Measured (core2, robinson, Agg): 5 chunks 4.65 s (25% inside draw_idle), 13 chunks 6.12 s (41%), 49 chunks 12.08 s (70%). **Applied:** new helper `do_progressive_draw()` gates all 5 call sites on an interactive backend (qt/tk/gtk/wx/macosx/nbagg/webagg/ipympl -> yes; agg/inline/pdf/svg -> no). Runtime is now flat in the chunk count: 3.9/3.2/3.3/3.5 s for 1/5/13/49 chunks (3.4x at 49 chunks), and the rendered PNG is pixel-identical between 1 and 49 chunks.

### 37. `pyvista`/`vtk` and `ipywidgets` imported eagerly at package import — ✅ FIXED

Both were paid by every `import tripyview` — in each papermill kernel and each dask worker — although most runs never touch 3D rendering or the interactive point picker. **Applied:** `ipywidgets` is imported inside `select_scatterpts_depth.__init__` (its only user, all 18 uses), and `sub_3dsphere` is imported on first attribute access via a module-level `__getattr__` (PEP 562) in `__init__.py` instead of `from .sub_3dsphere import *`; `tpv.create_3dsphere_*(...)` keeps working unchanged, the import just happens at that moment and is cached. `importlib.import_module` is used there because `from . import sub_3dsphere` re-enters `__getattr__` and recurses. `TRIPYVIEW_WITHOUT_VTK=1` behaves as before. Import time 4.1 s -> ~3.1 s; `pyvista`/`vtk`/`ipywidgets` are no longer in `sys.modules` after a plain import. No package module imports `sub_3dsphere`, and nothing uses `from tripyview import *` (which module `__getattr__` would not serve).

## Still open

- **Tests:** CI only runs `import tripyview`. A small pytest smoke suite (box masks, shapefile masks, chunk defaults, an image check of plot layering) would have caught #25-29 and the four zorder layering bugs.
- **Dependency check:** `setup.py` lists `libnetcdf`, which on PyPI is only a 0.0.1 placeholder, not the netCDF C library (that comes from conda).
