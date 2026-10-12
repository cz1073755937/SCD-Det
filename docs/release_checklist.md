# GitHub release checklist

- [x] Set the README clone URL to `cz1073755937/SCD-Det`.
- [x] Verify author order and corresponding-author contact against the published article.
- [x] Add the published paper's DOI and bibliographic metadata to README and `CITATION.cff`.
- [ ] Run `python -m pytest -q`.
- [ ] Apply the integration patch to a clean Ultralytics 8.3.175 checkout.
- [ ] Build `models/scd_det_n.yaml` and verify the expected model summary.
- [x] Search for private paths, credentials, private URLs, and dataset filenames.
- [x] Run `python tools/check_release.py` and resolve every reported item.
- [x] Confirm no `.pt`, dataset, run directory, cache, Word/PDF review draft, or internal log is staged.
- [ ] Add only audited result tables and clearly label one-seed versus mean±standard-deviation values.
- [ ] Tag the first immutable release (for example, `v0.1.0`) and archive its environment.
