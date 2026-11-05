# FRC Perception for REBUILT 2026

Jetson-based FUEL detection, stereo distance, tracking, clustering, and NetworkTables publishing for the 2026 FRC game. The end-to-end design, implementation notes, calibration caveats, and experiment log are in [FRC_PERCEPTION.md](FRC_PERCEPTION.md).

## Deploy

The service is configured for a 3200×1200 side-by-side MJPEG stereo camera. Review `deploy/frc-perception.env.example`, select the correct camera calibration, then install and start `deploy/frc-perception.service` on the Jetson. The service is intentionally not enabled automatically. The detector checkpoint is `models/rebuilt_fuel_yolo26n.pt`; build a Jetson-specific TensorRT engine with `scripts/build-engine.sh`.

NetworkTables output is disabled until `FRC_NT_ENABLED=1` and the team/server settings are configured. Topics and coordinate semantics are documented in `FRC_PERCEPTION.md`. Coordinates are camera-relative; no robot mounting transform is assumed.

## Replay and benchmark

Replay/evaluation scripts are under `scripts/`. The `benchmarks/live-usb-dual-pipeline-2026-11-05/` directory contains the two-pipeline live USB throughput profile and raw inference/NT traces. The test duplicated one stereo USB feed into two processing pipelines; it measures compute load, not two independent camera sensors or synchronization.
