# Fix Plan: mly_metadata.py Performance + Python 3.11 CI

Branch: `fix/mly-perf-py311`

---

## Problem 1 — `mly_metadata.py` slow / crashes

### Root-cause analysis

| Symptom | Location | Root cause |
|---|---|---|
| Hangs for hours at image level | `_ensure_street_network` → `ox.graph_from_bbox` | No progress feedback; downloads entire OSM network silently |
| 16× slow H3 computation | `_compute_h3_id_metadata_image` | `map_elements` (Python UDF) called N × 16 times |
| Double daytime/nighttime work (grid/street) | `_compute_number_of_daytime/nighttime_metadata_grid_street` | `_day_or_night(self.joined)` called twice independently |
| 4× redundant season group_by (grid/street) | `_compute_number_of_{season}_metadata_grid_street` | Separate `group_by().agg()` for each season |
| Crash `ColumnNotFoundError: organization_id` (unit=street) | `_compute_number_of_organizations_metadata_grid_street` | Assumes `organization_id` always present; Mapillary data often omits it |

### Fixes (all in `src/zensvi/metadata/mly_metadata.py`)

- **1a — H3 IDs**: Compute H3 on unique `(lat, lon)` pairs only, join back.
  Reduces Python UDF calls from `N × 16` to `unique_pairs × 16`.

- **1b — Daytime/nighttime cache**: Add `_get_joined_daynight()` helper that
  computes `_day_or_night(self.joined)` once and caches on `self._joined_daynight_cache`.
  Both daytime and nighttime methods use the cache. Cache is cleared at the start
  of each `compute_metadata` call.

- **1c — Seasons single group_by**: Add `_compute_all_seasons_cache()` that runs
  one `group_by().agg()` computing all four seasons. Four individual season methods
  draw from the cache. Cache cleared at start of `compute_metadata`.

- **1d — Street network progress**: Before `ox.graph_from_bbox` in
  `_ensure_street_network`, print/log a clear message so users know the download
  is in progress and how large the bounding box is.

- **1e — Optional organization_id**: In `_compute_number_of_organizations_metadata_grid_street`,
  check if `organization_id` is present in `self.joined.columns`. If missing,
  fill `number_of_organizations` with null instead of crashing.

---

## Problem 2 — Python 3.11 CI failure (`NameError: name 'nn' is not defined`)

### Root-cause analysis

The installed `transformers` version requires `torch >= 2.4`. With
`torch = ">=2.3,<2.10"` in `pyproject.toml`, CI resolves `torch==2.3.1` on Python 3.11.
`transformers/integrations/accelerate.py` then uses `nn.Module` in a type annotation
without importing `torch.nn as nn` (because PyTorch was disabled), causing
`NameError` that makes ALL 15 test files fail during collection.

### Fix (in `pyproject.toml`)

Bump `torch = ">=2.3,<2.10"` → `torch = ">=2.4,<2.10"`.

PyTorch 2.4 supports Python 3.9–3.12, so no version matrix changes needed.

---

## Verification steps

1. Create isolated sandbox venv, install local package, run `tests/test_mly_metadata.py`
   against existing small test CSV (`tests/data/input/metadata/mly_pids.csv`).
2. Confirm all 6 mly_metadata tests pass.
3. Run `poetry lock` to ensure new torch constraint locks cleanly.
4. Push branch, open PR targeting `main`.

---

## Status tracker

- [x] Branch `fix/mly-perf-py311` created
- [x] plan.md written
- [x] Fix 1a — H3 unique pairs (`_compute_h3_id_metadata_image`)
- [x] Fix 1b — Daytime/nighttime cache (`_get_joined_daynight`)
- [x] Fix 1c — Seasons single group_by (`_get_seasons_grouped_cache`)
- [x] Fix 1d — Street network progress message (`_ensure_street_network`)
- [x] Fix 1e — organization_id optional handling
- [x] Fix 2  — `torch = ">=2.4,<2.10"` in pyproject.toml
- [ ] Sandbox env test (no polars-capable Python env found locally; validated via CI)
