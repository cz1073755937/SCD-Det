# Reproducibility when images cannot be released

MCDD and WDD are not bundled with this repository. As stated in the [published paper](https://doi.org/10.1007/s11554-026-01982-1), they are available upon reasonable request to the corresponding author, Rongfen Gong ([fx_gong@hotmail.com](mailto:fx_gong@hotmail.com)).

Code availability alone does not provide the private images or guarantee reproduction of their exact benchmark numbers. Dataset documentation can still provide:

- why the data cannot be shared (industrial confidentiality, ownership, or privacy);
- acquisition device and environment at a non-identifying level;
- image count, native resolution distribution, class list, and instance count per class;
- fixed train/validation/test counts and the rule or seed used to form each split;
- annotation format, box inclusion rules, treatment of truncation/occlusion, and quality-control procedure;
- object-scale, density, and overlap/occlusion statistics;
- all resizing, normalization, augmentation, and preprocessing settings;
- checkpoint-selection metric on validation and the one-time test evaluation protocol;
- an access statement, even if the answer is that external access is unavailable;
- results on at least one genuinely public dataset using a published or included fixed split.

If filename disclosure is acceptable, publish split manifests. If filenames are sensitive, publish stable salted hashes or only split-generation code and aggregate counts. Never publish reversible identifiers without permission.
