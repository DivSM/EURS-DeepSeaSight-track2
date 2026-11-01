# Weights - ACTION NEEDED before this repository is submittable

`export_onnx.py` expects **one** checkpoint file directly in this folder (`model_dir`): `student_B.pt`, a plain
`torch.save(model.state_dict(), ...)` of
`StudentNet(enc="lcnet_075.ra2_in1k", d=56, trunk_c=40, trunk_nb=2, trunk_cf=12)` (see `src/student_net.py`),
**with the calibrated per-class logit bias already written into its `bias` buffer** before saving (this is the
"`new_alt`" lineage, stage-B / "ckpt" checkpoint of the team's Track 2 training notebook - see METHOD.md for the
exact calibrated values, which `export_onnx.py` also carries as a documented fallback).

This file is **not included in this repository** - it must be exported from the team's Kaggle training-notebook
output. Because a `StudentNet` of this size is only ~0.78 M parameters (~3 MB as an fp32 `.pt` file), it is well
under GitHub's 100 MB per-file limit, so the **simplest option is to commit it directly**:

1. Download `student_B.pt` (or whatever the training notebook's output calls the stage-B / "ckpt" checkpoint of
   the `new_alt` lineage) from the Kaggle notebook's output files.
2. Copy it into this folder as `weights/student_B.pt` and commit it with `git add -f weights/student_B.pt`
   (small files are sometimes in `.gitignore`-like ignore lists by accident - check `git ls-files weights` shows
   it).
3. Delete `weights/weights_link.yaml` (committed weights and a link are not used together).
4. Test it: `python repro_check.py . --export-onnx`. It regenerates the ONNX file with `export_onnx.py` and
   reports whether it matches `onnx/submitted_model.onnx` (see `onnx/README.md` - that file also still needs to
   be added).

If a public link is preferred instead (for example because the weights are hosted on Hugging Face alongside the
`onnx/submitted_model.onnx` copy), fill in `weights/weights_link.yaml` with that URL and the file's SHA-256
instead, and keep this folder otherwise empty.

Until one of these is done, this repository will **not** pass `repro_check.py` or reproduce on the organizers'
side.
