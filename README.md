# EURS @ MaCVi WACV 2027 - Track 2 reproduction repository (team DeepSeaSight)

Reproduces testing-phase submission **#19180** (composite 0.385): a single-network student (`StudentNet`,
PP-LCNet encoder, FPN-lite segmentation head, physics-shaped restoration head, half-res detail trunk), exported
to a static-shape ONNX graph with a fixed box-blur unsharp step ("k3a2.5") appended. See **`METHOD.md`** for the
full method description, every external dataset / pretrained model used, and exactly which choices were fixed
in advance by testing-phase leaderboard score.

```
repro.yaml         team / track / submission_id / contact
METHOD.md          approach, training data, external data, non-standard inference details
export_onnx.py     main(model_dir, out_onnx) - the required entry point
src/               model code imported by export_onnx.py (StudentNet)
weights/           ACTION NEEDED - see weights/README.md: the trained checkpoint is not yet attached
onnx/              ACTION NEEDED - see onnx/README.md: a copy of the leaderboard .onnx file is not yet attached
repro_check.py     the organizers' checker; run before submitting
docker/Dockerfile  the organizers' base image
```

Before handing this in: add `weights/student_B.pt` (or fill `weights/weights_link.yaml`) and
`onnx/submitted_model.onnx` per their README files, then run `python repro_check.py . --export-onnx`.

`export_onnx.py` and `src/student_net.py` were verified end-to-end in this session with a randomly-initialised
checkpoint (the real trained weights are not available in this environment): the script builds the network,
exports a static `(1,3,448,448)` graph with outputs `restored_image`/`seg_logits`, appends the box-blur unsharp
step, passes `onnx.checker`, and runs correctly under `onnxruntime`.
