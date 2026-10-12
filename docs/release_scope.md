# Public release scope

## Publish now

- `scd_det/modules.py`: FREG, CEBA, and UMA source.
- `scd_det/bsc_loss.py`: paper BSC, class-wise BSC, and controlled alternatives.
- `models/scd_det_n.yaml`: the selected model graph and internal parameters.
- `patches/`: minimal Ultralytics 8.3.175 parser/loss integration.
- `configs/`, `data/`, and `tools/`: portable examples without machine-specific paths.
- Tests, dependency pins, license, citation metadata, and implementation details.

This is the minimum useful scientific release: readers can inspect every claimed module, see where BSC is calculated, construct the model, and train it on a compatible YOLO-format dataset.

## Publish after the experiment audit

- Final result tables with exact split, seed count, mean and standard deviation.
- Checkpoints whose training configuration and dataset authorization are verified.
- End-to-end timing tables with preprocessing, host-to-device transfer, inference, and postprocessing definitions.
- Sanitized qualitative examples for which publication rights are confirmed.
- A paper/preprint link after the publisher or preprint policy is confirmed.
- A public-dataset download/config helper only after its license and fixed split are verified.

## Do not publish

- MCDD or WDD images, labels, filenames, private storage paths, or archives.
- Private dataset access credentials or cloud links.
- Reviewer reports, response drafts, editor correspondence, or unpublished Word documents.
- Internal queue/audit scripts, machine management, GPU mirroring, process logs, and failure records.
- `runs/`, `runs_revision/`, TensorBoard data, caches, checkpoints, or temporary files.
- Machine-specific absolute paths, local usernames, tokens, or environment dumps.

## Important limitation

Releasing method code does not make private-data benchmark numbers independently reproducible. State this directly. The release supports implementation reproducibility; private benchmark reproducibility is limited to documented splits, annotation rules, statistics, and evaluation protocol.
