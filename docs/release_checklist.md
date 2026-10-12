# GitHub release checklist

- [ ] Replace `YOUR_ACCOUNT` in README after creating the repository.
- [ ] Confirm author names, ordering, repository owner, and contact information.
- [ ] Confirm manuscript/preprint citation and publisher sharing policy.
- [ ] Run `python -m pytest -q`.
- [ ] Apply the integration patch to a clean Ultralytics 8.3.175 checkout.
- [ ] Build `models/scd_det_n.yaml` and verify the expected model summary.
- [ ] Search for absolute paths, usernames, credentials, private URLs, and dataset filenames.
- [ ] Run `python tools/check_release.py` and resolve every reported item.
- [ ] Confirm no `.pt`, dataset, run directory, cache, Word/PDF review draft, or internal log is staged.
- [ ] Add only audited result tables and clearly label one-seed versus mean±standard-deviation values.
- [ ] Tag the first immutable release (for example, `v0.1.0`) and archive its environment.
