# Reproducibility when images cannot be released

MCDD and WDD remain private. The repository should not imply that third parties can reproduce their exact benchmark numbers. A useful and defensible data statement can still provide:

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
