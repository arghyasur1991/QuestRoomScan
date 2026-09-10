# Changelog

All notable changes to this package are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/); versions follow
[Semantic Versioning](https://semver.org/).

## [Unreleased]

### Live scan

- Body exclusion is capsules, not a 0.6 m head cylinder: torso (0.35 m,
  world-up), hands (0.14 m), and short forearms. Tests both the depth sample
  (a body pixel integrates nothing along its ray — the negative band behind a
  hand otherwise meshes as a hand-shaped shell) and the voxel (a wall behind
  a hand still fills). `FreezeInView` skips capsules; unfreeze does not.
  Optional `eraseBodyBlobs` (off) can clear leftover hand voxels below
  `eraseMaxWeight` (0.2, above seed weight). A 2 s diagnostic line logs the
  anchors and capsules while scanning.
- `DepthCapture.removeHandsFromDepth` requests Meta occlusion hand removal
  (inpaints depth). Needs hand tracking; the runtime turns it off while
  holding controllers. Capsules cover that case.
- Default: `RoomScanner` refreshes head / wrists from `OVRCameraRig` each
  integrate (controller → `HandOnControllerAnchor`; else tracked `OVRHand`).
  `RoomScanSession.SetBodyExclusionAnchors` is for hosts with a non-OVR
  rig; it marks anchors host-owned so the scanner will not overwrite them.
- Scan progress is analytic. `ScanProgress.OverallProgress =
  min(Closure, Refinement)`: closure from the live mesh's own boundary
  edges (Surface Nets records every crossing edge whose quad could not be
  emitted; a GPU connected-components pass over a snapshot clusters them
  into loops, drops loops on clip planes / the volume edge as cuts, ignores
  loops under `holeMinPerimeter`), refinement from the fraction of surface
  voxels at or above `confidentWeight`. Time-sliced over ~16 frames, one
  128-byte readback per `analysisIntervalSeconds`. `ScanCoverage` gains
  `AnalysisAvailable`, `Closure`, `Refinement`, `ConfidentSurfaceCount`,
  `OpenBoundaryMetres`, `HoleCount`, `LargestHole`. Needs no scene model.
- Removed: the frozen-fraction / colour / plateau progress blend,
  `ScanCoverage.IsStabilized`, `coverageUpdateInterval`. `FrozenFraction`
  and `ColorCoverage` stay as raw fields.
- Shell coverage (needs `RoomUnderstanding`): the captured hull — outer
  walls, floor, ceiling, furniture faces — is sampled into ≤ 16k cells and
  marched against the TSDF at the analysis tick. `ScanCoverage` gains
  `ShellCoverage` (openings excluded from the denominator), `ShellGapCount`,
  `LargestGap`, `ShellFillsApplied`. Guidance and auto-fill only; it never
  feeds `OverallProgress`. `RoomScanSession.CopyShellGaps` lists the
  largest unscanned patches.
  Furniture faces march through the whole scene box, and a segment the
  sensor has seen straight through (observed free space) is reported empty
  and leaves the denominator (`ShellCellsEmpty`) — air inside a loose couch,
  bed, or table box is not a hole.
- Auto-fill while scanning: small wall / floor / ceiling gaps whose covered
  neighbours lie on one plane are stamped with that plane at a soft weight
  (`autoFillShellGaps`, on); small furniture gaps get a cluster-local
  6-neighbour close (`closeFurnitureHoles`, on). Real depth overrides both.
  No volume-wide pass is added.

## [1.0.0] - 2026-09-09

First stable release. `RoomScanSession` is API-stable from here; breaking
changes bump the major version. `v0.1.0` predates almost everything below,
so this entry describes the package rather than a diff.

### Live scan (GPU only, zero CPU readback)

- TSDF volume integration from the Quest depth sensor (256³, RG8 SNorm +
  RGBA8 colour): bilateral depth filter, dilation, normals, adaptive
  weighting, pruning, exclusion zones around tracked heads.
- GPU Surface Nets mesh extraction in compute, drawn with one
  `Graphics.RenderPrimitivesIndirect`; adaptive per-vertex temporal damping,
  HC Laplacian smoothing, plane-snap regularization. Indirect argument
  buffers are zeroed at allocation so the preview draws nothing until the
  first extraction.
- Two-layer texturing: triplanar world-space colour cache (~8 mm/texel) from
  the passthrough RGB camera, vertex colour fallback. Triplanar optional.
- Presentation-only live-mesh birth fade and hold-and-morph between
  extractions (`GPUVertex` 48 bytes with `prevPos`; authoring paths read
  extractor `pos` via `ExtractForAuthoring`).
- `FreezeInView` / `UnfreezeInView`, `ScanCoverage` / `ScanProgress`,
  render modes Wireframe / Vertex / Triplanar / Refined / Occlusion / Splat /
  None, freeze tint toggle.
- Defaults tuned in a shipped title: extract 8 Hz, keyframes 0.4 m / 20° /
  1 s, post-bake simplify 0.5, vertex budget 8 %, warmup 3.

### Scene API priors (optional `RoomUnderstanding`)

- `ConfineScanToContainingRoom` (default off): TSDF clipped to the MRUK room
  containing the headset — outer walls / floor / ceiling expanded 50 cm
  outward, then hard-confined, with a matching AABB cull; triplanar bake
  honours the same clip.
- `StampScreenPlanes` (default on): MRUK `SCREEN` planes written as analytic
  TSDF slabs via a voxel-AABB dispatch so TV glass is not depth noise.
- Without the module the scan is unbounded and occupancy APIs return
  false / empty; permissions and MRUK load do not depend on it.

### Refinement and persistence

- Texture refinement: xatlas UV unwrap (native plugin; builds on macOS /
  Windows / Linux hosts), multi-view atlas bake from motion-gated keyframes,
  Sobel normal maps, UV-preserving post-bake simplification (meshoptimizer).
- Package-based persistence: `pkg_YYYYMMDD_HHMMSS/` with TSDF, triplanar,
  keyframes, refined mesh + atlas, splat, and a manifest; `_tmp/` staging;
  `LoadRefinedOnlyAsync` loads mesh + atlas in under a second.
- `OVRSpatialAnchor` relocation per package with per-artifact creation
  matrices; each package stores the Scene API UUID of the room it was
  scanned in, rebound from the anchor pose when missing or stale.
- `RoomSpaceRoot`: content parented under it keeps room-registered local
  coordinates across sessions (`WaitForBindAsync`, `WorldToRoom`, `Adopt`,
  `AdoptAtRoomOrigin`).

### Game integration — `RoomScanSession`

- `StartScanAsync` → `FreezeInView` → `FinalizeScanAsync` → `ScanResult`;
  `LoadAsync` / `LoadLatestAsync` / `LoadRefinedOnlyAsync`;
  `ListSavedScans` / `DeleteScanAsync` / `UnloadActiveScanAsync` /
  `ClearAllScansAsync`; `RefinedMesh` / `RefinedAtlas` / `RefinedMeshRenderer`.
- One serialised permission queue (`AndroidRuntimePermission`): one dialog at
  a time, same-permission callers share a task. `StartScanAsync` requests
  what is missing before bring-up — `USE_SCENE` (required), then
  `HEADSET_CAMERA` / `USE_ANCHOR_API` (degrade). Hosts may front-load via
  `Request{Scene,Camera,Anchor}PermissionAsync` / `Has*Permission`.
- Scene-room occupancy from outer wall planes, never `GetCurrentRoom()`:
  `IsHeadsetInsideASceneRoom`, `IsHeadsetInsideBoundSceneRoom`,
  `HeadsetSceneRoomUuid`, `BoundSceneRoomUuid`,
  `TryRebindBoundSceneRoomIfHeadsetMatches`, `CopyHeadsetRoomWallFaces`
  (`SceneWallFace`, `IsScreen` for TVs).
- Discovery control: `WaitUntilRoomReadyAsync`, `IsRoomLoaded`,
  `HasSceneRooms`, `ReloadSceneFromDeviceAsync`,
  `RequestSpaceSetupAndReloadAsync`. `RoomReady` always signals (zero rooms,
  denied permission, missing `SceneLoadedEvent`); scene load never
  auto-launches Space Setup.
- Depth sensor and RGB camera run only during a scan; both are disabled at
  `Awake` and by the wizard.

### Beyond the mesh

- Gaussian Splat pipeline: keyframes + PLY point cloud → server training →
  on-device rendering (`HAS_GAUSSIAN_SPLATTING`).
- Optional YOLO object detection via Unity Inference Engine with GPU NMS
  (`HAS_AI_INFERENCE`), projected to 3D and merged with MRUK anchors in a
  `SceneObjectRegistry`; debug visualizer.
- Server-side atlas and mesh enhancement clients.

### Tooling

- Game-ready setup wizard: Meta Building Blocks, URP pipeline, build profile,
  VR UI input pipeline, `AndroidManifest`, shader wiring, xatlas build,
  scan defaults; turns `OVRManager`'s startup permission dialog off.
- Two-panel world-space VR debug menu: scan, saved-scan browser, refine,
  Gaussian Splat, tools.
- Meta XR SDK 205, Unity 6; every shader compiles warning-free on Metal and
  Vulkan.

## [0.1.0] - 2026-03-08

Initial tag: GPU TSDF volume integration from Quest depth, CPU mesh
extraction, triplanar world-space texture cache, plane detection and
4-phase mesh regularization, fast-convergence stability overhaul.
