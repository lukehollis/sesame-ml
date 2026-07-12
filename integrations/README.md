# Vendor-side integration templates

These files are copied into the exact vendor checkout used for fine-tuning. Runtime clients
live in `src/sesame_ml/integrations/`; these templates configure the vendor model/dataset side.

- `openpi/`: pi0, pi0-FAST and pi0.5 physical-eight-dimensional Sesame transforms/config
  factories. pi0/pi0.5 use the upstream 32D checkpoint head with explicit zero-padding;
  pi0-FAST remains 8D. Checked against OpenPI commit
  `15a9616a00943ada6c20a0f158e3adb39df2ccac`.
- `groot/`: GR00T N1.7 `NEW_EMBODIMENT` modality/config, checked against Isaac-GR00T commit
  `9c7e746b2cd37a810070a98ef41d290a07e806c2`.

The GR00T Python configuration preserves NVIDIA's Apache-2.0 SPDX attribution. OpenPI-derived
configuration/transform patterns are Apache-2.0. Review both templates against the exact
upstream commit installed; vendor APIs may move.

The full simulator dependency graph is intentionally not installed into a vendor environment.
Run `python -m sesame_ml.bridge_cli` with this repository's `src/` on `PYTHONPATH` and preserve
each vendor's frozen lock. `bridge-requirements.txt` is for a dedicated exporter or ephemeral
overlay, as documented in `docs/vla-finetuning.md`.

The same files are packaged under `sesame_ml/vendor_templates/` so wheel users can locate them
with `importlib.resources.files("sesame_ml.vendor_templates")`.
