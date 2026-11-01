# EURS @ MaCVi - Track 2 reproduction repository 
Team: OffshorePixels   Method: DeepSeaSight

Submission **#19180** (composite 0.385)
A single-network student (`StudentNet`, PP-LCNet encoder, FPN-lite segmentation head, physics-shaped restoration head, half-res detail trunk), 
exported to a static-shape ONNX graph with a fixed box-blur unsharp step ("k3a2.5") appended. 
See **`METHOD.md`** for the full method description, every external dataset / pretrained model used, and exactly which choices were fixed in advance.

```
repro.yaml         team / track / submission_id / contact
METHOD.md          approach, training data, external data, non-standard inference details
export_onnx.py     main(model_dir, out_onnx) - the required entry point
src/               model code imported by export_onnx.py (StudentNet)
weights/           trained checkpoint is attached
onnx/              .onnx file is attached
repro_check.py     the organizers' checker
docker/Dockerfile  the organizers' base image
```
