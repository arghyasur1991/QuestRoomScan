# Changelog

All notable changes to this package are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/); versions follow
[Semantic Versioning](https://semver.org/).

## [Unreleased]

## [1.1.0] - 2026-09-11

Analytic scan progress, body exclusion that holds up, and a texture bake
that ignores the player's hands. `RoomScanSession` only gains members;
`ScanCoverage` loses its legacy stabilisation fields (see Removed).

### Added

- **Analytic scan progress** from the boundary of observed free space. Every
  TSDF voxel is observed-free, observed-solid or unknown; unknown that
  touches a full 8³ block of unknown (a 40 cm cube of nothing — the outside,
  the far side of a hole, the inside of a couch) is *void*, carried through
  the shell blocks by an LDS fine flood. Free–solid faces are the surface,
  free–void faces are **leaks** — exactly where passthrough shows through the
  mesh. Faces within `cutToleranceVoxels` of a clip plane or the volume edge
  are cuts, not leaks. `Closure = surface / (surface + leak)`;
  `ScanProgress.OverallProgress = Closure × (1 − refinementInfluence × (1 −
  Refinement))`, refinement being the fraction of surface voxels at or above
  `confidentWeight`. One full-volume classify per cycle, everything else over
  the ~10 % of blocks that hold both unknown and observed voxels, time-sliced
  over ~20 frames, one 128-byte readback per `analysisIntervalSeconds`. No
  camera pose, no rays, no scene model.
  `ScanCoverage` gains `AnalysisAvailable`, `Closure`, `Refinement`,
  `ConfidentFraction`, `ConfidentSurfaceCount`, `LeakAreaM2` (the absolute
  number a host should gate on), `SurfaceAreaM2`, `HoleCount`, `LargestHole`
  (`MeshHole`: centre, area, faces), `LeakFills`.
- **Leak tint** (`RoomScanSession.ShowHoles`): the labels live in an
  `R8_UNorm` 3-D texture (`gsLabelVolume`) the scan mesh shader samples per
  vertex, so what is red is what is counted, listed and filled.
- **Leak fill** (`fillLeaks`, on): leak patches up to `fillLeakMaxAreaM2`
  (0.25 m²) are capped by turning the two unknown voxels behind each leak
  face solid, continued from the free voxel's own TSDF value at the band
  slope, so a hole in a wall closes on the wall. Frontier-sized patches are
  never capped; real depth is never overwritten.
- **Body exclusion capsules** replace the 0.6 m head cylinder: torso
  (0.35 m, world-up), hands (0.14 m) and short forearms. Tested at both the
  depth sample (a body pixel integrates nothing along its ray — the negative
  band behind a hand otherwise meshes as a hand-shaped shell) and the voxel
  (a wall behind a hand still fills). `FreezeInView` skips capsules; unfreeze
  does not. Optional `eraseBodyBlobs` (off) clears leftover hand voxels
  below `eraseMaxWeight`. `RoomScanner` refreshes head / wrists from
  `OVRCameraRig` each integrate (controller → `HandOnControllerAnchor`, else
  tracked `OVRHand`); `RoomScanSession.SetBodyExclusionAnchors` is for hosts
  with a non-OVR rig and marks the anchors host-owned.
- `DepthCapture.removeHandsFromDepth` requests Meta occlusion hand removal
  (inpainted depth). Needs hand tracking; the runtime turns it off while
  controllers are held, which the capsules cover.
- **Hands stay out of the texture.** `KeyframeCollector` records the hand /
  forearm capsules at capture (`"cap"` in `frames.jsonl`, relocated with the
  pose on load) and skips frames where they cover more than
  `maxHandCoverage` (12 %) of the image; `AtlasBakeCompute` rejects any
  texel whose line of sight from that view passes through a recorded
  capsule (radius × 1.4), in both the best-view and the blend pass.
- **Shell coverage** (needs `RoomUnderstanding`): the captured hull — outer
  walls, floor, ceiling, furniture faces — is sampled into ≤ 16k cells and
  marched against the TSDF on the analysis tick. `ScanCoverage` gains
  `ShellCoverageAvailable`, `ShellCoverage` (openings excluded from the
  denominator), `ShellCellsTotal / Covered / Excluded / Empty`,
  `ShellGapCount`, `LargestGap`, `ShellFillsApplied`;
  `RoomScanSession.CopyShellGaps` lists the largest unscanned patches.
  Furniture faces march through the whole scene box, and a segment the
  sensor has seen straight through is reported empty and leaves the
  denominator — air inside a loose couch or table box is not a hole.
  Guidance and auto-fill only; it never feeds `OverallProgress`.
- **Shell auto-fill** while scanning: small wall / floor / ceiling gaps whose
  covered neighbours lie on one plane are stamped with that plane at a soft
  weight (`autoFillShellGaps`, on); small furniture gaps get a cluster-local
  6-neighbour close (`closeFurnitureHoles`, on). Real depth overrides both.
- **Freeze / unfreeze spotlight**: `FreezeInView` / `UnfreezeInView` paint a
  head-forward cone (`freezeConeHalfAngle`, 15°) instead of the whole depth
  frustum; `RoomScanSession.FreezeConeHalfAngle` lets a host draw the ring.

### Changed

- Multi-view bake: `blendMinFraction` 0.3 → 0.75 and new `maxViewsPerTexel`
  (3), so a long scan no longer averages dozens of misregistered views into
  mush. Keyframe capture thresholds 0.4 m / 20° → 0.5 m / 25°.
- `ScanProgress.Phase` thresholds now read `OverallProgress` (< 0.30
  Discovering, < 0.90 Refining, < 0.95 Stabilized, else Complete).
- The compute package compiles warning-free on Vulkan (single-exit helpers,
  direction tables instead of dynamic vector-component writes).

### Removed

- The frozen-fraction / colour / vertex-plateau progress blend:
  `ScanCoverage.IsStabilized`, `VolumeIntegrator.coverageUpdateInterval` and
  the separate coverage readback. `FrozenFraction` and `ColorCoverage` stay
  as raw fields. Hosts that gated on `IsStabilized` should gate on
  `Coverage.LeakAreaM2` (absolute) or `OverallProgress`.
- `RoomScanner.TryGetCameraIntrinsics` and the body-exclusion diagnostic log.

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
