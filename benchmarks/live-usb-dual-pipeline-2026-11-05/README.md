# Live USB dual-pipeline benchmark — 2026-11-05

One 3200×1200 MJPEG stereo USB capture was fanned out to two independent perception pipelines. Each measured about 52 input frames/s, 14 inference updates/s, and 17.5–17.9 optical-flow updates/s. Ingress-to-NT-publish latency was 105–109 ms p50 and 139–144 ms p95. Stereo pairs were complete.

This duplicates one scene into two compute pipelines; it does not model two cameras' USB bandwidth or synchronization. Latency starts at each fan-out client ingress and excludes sensor exposure and upstream fan-out transport. The trace files are raw local-pipeline measurements; the profile JSON contains the measurement window and summarized counters.
