# onnx/ - ACTION NEEDED before this repository is submittable

Put a copy of the exact file uploaded to the leaderboard for the reported submission (#19180,
`T2_01_new_alt_k3a2.5.onnx`, testing-phase composite 0.385) here, renamed to:

    onnx/submitted_model.onnx

Then run `python repro_check.py . --export-onnx`: it regenerates the file with `export_onnx.py` and reports
whether it equals this copy (byte-identical, or the same graph and weights with only the file header, such as
the torch/opset version, different - that is fine). Write the file's SHA-256 (`sha256sum onnx/submitted_model.onnx`)
into the factsheet.

`export_onnx.py` was verified in this session (with a randomly-initialised checkpoint, since the real trained
weights are not available in this environment) to produce a graph with: static input `(1, 3, 448, 448)`, outputs
`restored_image` then `seg_logits`, ~0.79 M parameters, and an `onnx.checker`-valid, `onnxruntime`-runnable file
containing the appended box-blur unsharp step - so once the real `weights/student_B.pt` is in place, this script
is expected to reproduce the real file closely; only the trained numbers will differ from this placeholder test.
