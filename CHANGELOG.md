# Changelog

All notable changes to this package are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/); versions follow
[Semantic Versioning](https://semver.org/).

## [Unreleased]

## [1.0.0] - 2026-09-09

First stable release. The game-integration surface (`RoomScanSession`) is
considered API-stable from here; breaking changes bump the major version.

`v0.1.0` was tagged before most of the package existed. Everything below is
new since that tag; the first block landed on `main` between March and
August 2026 (PRs #2–#19), the rest is this release's `develop` → `main`.

### Landed on `main` since 0.1.0 (PRs #2–#19)

- GPU Surface Nets meshing — zero-readback compute extraction, one
  `RenderPrimitivesIndirect` draw; CPU meshing path removed. Integration and
  extraction moved to 30 Hz.
- Adaptive per-vertex temporal damping, HC Laplacian smoothing, plane-snap
  regularization.
- Texture refinement: xatlas UV unwrap (native plugin, builds on macOS /
  Windows / Linux hosts), multi-view atlas bake, Sobel normal maps,
  UV-preserving post-bake simplification via meshoptimizer; server-side atlas
  / mesh enhancement.
- Package-based persistence (`pkg_YYYYMMDD_HHMMSS/`, `manifest.json`,
  `_tmp/` staging), `OVRSpatialAnchor` relocation with per-artifact creation
  matrices, forward-splat triplanar relocation, `LoadRefinedOnlyAsync`.
- Gaussian Splat pipeline: motion-gated keyframes, PLY point-cloud export,
  server training client, on-device UGS rendering (gated behind
  `HAS_GAUSSIAN_SPLATTING`).
- MRUK `RoomUnderstanding` + `SceneObjectRegistry`, optional YOLO object
  detection via Inference Engine with GPU NMS (`HAS_AI_INFERENCE`), scene
  object debug visualizer.
- Render modes: Wireframe / Vertex / Triplanar / Refined / Occlusion / Splat /
  None; freeze tint toggle; `FreezeInView` / `UnfreezeInView`.
- `RoomScanSession` game facade (`StartScanAsync` → `FinalizeScanAsync` →
  `ScanResult`, `LoadLatestAsync`, `ClearAllScansAsync`, `ScanCoverage` /
  `ScanProgress`), VR debug menu, game-ready setup wizard (Building Blocks,
  URP, build profile, VR input pipeline, `AndroidManifest` checks).
- Depth subsystem gating, deferred ~600 MB scan GPU allocation until the first
  scan / save / full load (#17), superseded-anchor child detach (#18),
  `RoomSpaceRoot` anchor-relative persistence, floor-plane normal fix and
  Meta XR SDK 205 (#19).

### Fixed

- **Scan-start hang.** `GPUSurfaceNets` allocated its indirect draw / dispatch
  argument buffers with `new GraphicsBuffer(...)` and never wrote them from the
  CPU; the live vertex-mode preview drew from them every frame from scan start
  until the first extraction filled them. `GraphicsBuffer` contents are not
  zero-initialised on Vulkan, so a garbage index count had the GPU drawing
  millions of triangles a frame — measured as OS GPU pressure pinned at max,
  ~10 s frames and fence resets, recovering only when depth arrived 30–70 s
  later. Whether the garbage was zero depended on what the host app had just
  freed, which made the hang look flow-dependent. Both buffers are now zeroed
  at allocation.
- `RoomAnchorManager.RoomReady` / `IsRoomLoaded` now always signal, including
  when discovery finds zero rooms, the permission is denied, or MRUK never
  raises `SceneLoadedEvent` — hosts are no longer stuck waiting.
- The `Integrate` kernel reads the volume only after the analytic frustum
  rejections; two redundant voxel re-snap lines removed.

### Added

- **Multi-package session API** on `RoomScanSession`: `ListSavedScans()`,
  `DeleteScanAsync(id)`, `UnloadActiveScanAsync()` (drops the in-memory mesh
  and spatial-anchor bind without touching saved packages). A non-resume
  `StartScanAsync` unloads first.
- **One serialised permission queue.** Android drops a second
  `RequestUserPermission` while another dialog is in flight — no UI, no
  callback — and the package used to have three independent requesters
  (`OVRManager` at startup, `DepthCapture` at `Start`, the host). Every
  request now goes through `AndroidRuntimePermission`, which opens one
  dialog at a time and shares a pending task between callers asking for
  the same permission. `RoomScanner.StartScanningAsync` requests whatever
  is still missing before any GPU bring-up — `USE_SCENE` (required; a
  denial aborts the start), then `HEADSET_CAMERA` and `USE_ANCHOR_API`
  (a denial degrades) — so a bare `StartScanAsync()` always gets its
  dialogs. Hosts may front-load via `RequestScenePermissionAsync` /
  `RequestCameraPermissionAsync` / `RequestAnchorPermissionAsync`
  (+ `HasScenePermission` / `HasAnchorPermission`) for their own UX; those
  join the same queue and make the scan-start requests free. `DepthCapture`
  observes `USE_SCENE` and never requests. The setup wizard now turns
  `OVRManager.requestPassthroughCameraAccessPermissionOnStartup` **off**
  (it requested outside the queue).
- **Scene-room occupancy** (`RoomUnderstanding`, surfaced on the session):
  `IsHeadsetInsideASceneRoom`, `IsHeadsetInsideBoundSceneRoom`,
  `HeadsetSceneRoomUuid`, `BoundSceneRoomUuid`,
  `TryRebindBoundSceneRoomIfHeadsetMatches`, `CopyHeadsetRoomWallFaces`
  (`SceneWallFace`, with `IsScreen` for televisions). Occupancy walks every
  loaded room's outer wall planes (doorway faces included) — never
  `GetCurrentRoom()`, which is last/first after you leave, and not the floor
  outline alone, which stays true past a door.
- Each package stores the **Scene API UUID of the room it was scanned in**
  (`sceneRoomUuid` in `anchor.json` and the manifest), rebound from the
  localized anchor pose when missing or stale.
- **Scene discovery control:** `WaitUntilRoomReadyAsync`, `HasSceneRooms`,
  `ReloadSceneFromDeviceAsync` (re-run discovery after a late permission
  grant), `RequestSpaceSetupAndReloadAsync`. `LoadSceneFromDevice` runs with
  `requestSceneCaptureIfNoDataFound: false` so a missing scene model no longer
  pauses the app into Horizon Space Setup uninvited.
- **Scan priors** (need `RoomUnderstanding`): `ConfineScanToContainingRoom`
  (default off) clips TSDF to the room that contained the headset at scan
  start — outer walls / floor / ceiling expanded 50 cm outward, then
  hard-confined, with a matching AABB cull; `StampScreenPlanes` (default on)
  writes MRUK `SCREEN` planes as analytic TSDF slabs via a dedicated
  voxel-AABB dispatch so TV glass does not become depth noise. The triplanar
  bake honours the same clip.
- **Live-mesh presentation:** birth fade (cyan→photoreal settle with a short
  grow along the normal) and hold-and-morph between extractions, both in
  the forward shader only. `GPUVertex` is now 48 bytes (`prevPos` added);
  unwrap, atlas bake and PLY export go through `MeshExtractor.ExtractForAuthoring`
  and read extractor `pos`, never the in-flight lerp.
- `RoomScanner.RefinedMeshRenderer` / `RoomScanSession.RefinedMeshRenderer`.
- `MeshExtractor.TryExtract` — skips a dump while the previous one is still
  morphing on screen.

### Changed

- **Depth sensor and RGB camera run only during a scan.** `AROcclusionManager`
  and `PassthroughCameraAccess` are disabled at `Awake` and by the setup
  wizard; `StartScanningAsync` / `StopScanning` bracket them. The startup
  "briefly enable to verify" pass is gone.
- Game-proven scan defaults: `meshExtractionHz` 30 → **8**, keyframe gate
  0.15 m / 10° / 0.25 s → **0.4 m / 20° / 1 s**, `postBakeSimplificationRatio`
  1.0 → **0.5**, `gpuVertexBudgetPercent` 0.05 → **0.08**,
  `warmupIntegrations` 15 → **3**. The wizard's Game-Ready Apply restamps
  these onto scenes serialized against the old values.
- Temporal blend always runs (it also stamps birth-fade ticks); `_TemporalState.w`
  packs fade ticks and stability age.
- `RoomUnderstanding` is documented as optional: without it the scan is
  unbounded (no clip, no SCREEN stamps) and occupancy APIs return false /
  empty.
- Corrected the explanation of why `StartScanningAsync` is staged across
  frames. Earlier docs attributed a scan-start hang to PCA's MRUK handshake
  racing compute dispatches; the measured cause was the uninitialised
  indirect-args buffer above. The staging remains because it spreads the
  ~600 MB bring-up across frames.

### Removed

- `DepthCapture` and `PassthroughCameraProvider.StartCapture` no longer call
  `RequestUserPermission` directly.
- `RoomUnderstanding` no longer falls back to `Rooms[0]` when the headset is
  in no captured room.

## [0.1.0] - 2026-03-08

Initial tag: GPU TSDF volume integration from Quest depth, CPU mesh
extraction, triplanar world-space texture cache, plane detection and
4-phase mesh regularization, fast-convergence stability overhaul.
