# QuestRoomScan — Algorithm Reference

## Overview

Real-time 3D room reconstruction on Quest 3 using a TSDF (Truncated Signed Distance Field) volume and fully GPU-driven Surface Nets mesh extraction.

```
DepthCapture (AR depth frames → normals → dilation)
       │
VolumeIntegrator (TSDF integrate → warmup clear → prune)
       │
MeshExtractor → GPUSurfaceNets (compute shader: classify → compact → smooth → snap → temporal → index)
       │         └── GPUMeshRenderer (Graphics.RenderPrimitivesIndirect, single draw call)
       │
       ├── PlaneDetector (periodic RANSAC on background thread → persistent plane list)
       ├── TriplanarCache (bake camera → 3 world-space textures, persistent)
       └── KeyframeStore (ring buffer of camera frames, live quality)
                │
ScanMeshVertexColor.shader (SV_VertexID + StructuredBuffer → keyframes → triplanar → vertex color)
```

## 1. Volume Layout

### TSDF Volume
- **Format:** `R8G8_SNorm` — 2 bytes per voxel
  - **R channel:** Signed distance to surface, normalized by truncation distance. Range [-1, 1]. Negative = inside surface, positive = outside.
  - **G channel:** Confidence weight [0, 1]. Tracks how many quality observations support this voxel's value.
- **Dimensions:** 160 × 128 × 160 voxels (default)
- **Voxel size:** 0.05 m
- **World coverage:** 8m × 6.4m × 8m, centered at origin
- **Memory:** ~6.5 MB
- **Empty marker:** R = `sbyte.MinValue` (-128), G = 0

### Color Volume
- **Format:** `R8G8B8A8_UNorm` — 4 bytes per voxel
  - **RGB:** Accumulated camera color (exposure-boosted, quality-weighted running average)
  - **A:** Coverage weight [0, 1]. Tracks color confidence.
- **Dimensions:** Same as TSDF
- **Memory:** ~13 MB

### Coordinate System
- Voxel `(0,0,0)` maps to world `(-VoxelCount/2 * voxelSize)`.
- `WorldToVoxel(pos)`: `floor(pos / voxelSize + VoxelCount / 2)`, clamped.
- `VoxelToWorld(idx)`: `(idx + 0.5 - VoxelCount / 2) * voxelSize`.

## 2. Integration Pipeline

### Frustum Volume Construction
Built once at startup from the depth camera's projection matrix:
1. Decompose depth projection into frustum planes
2. Iterate a 3D grid from `zNear` to `maxUpdateDist` with step `voxelSize`
3. For each grid cell, compute view-space position `(x, y, -z)`
4. Include if `minUpdateDist < distance < maxUpdateDist`
5. Store as `ComputeBuffer` of `float3` positions (view space)
6. Cap at `maxFrustumPositions` (default 1M)

### Per-Voxel Integration (Integrate kernel)
For each frustum position (dispatched as 1D, 64 threads/group):

**Step 1: World position**
```
vLocalPos = frustumVolume[id]           // view space
vWorldPos = depthViewInv * vLocalPos    // world space
coord = worldToVoxel(vWorldPos)         // voxel indices
voxPos = voxelToWorld(coord)            // snapped world position
```

**Step 2: Early rejections**
- Behind camera: `voxView.z > -0.05`
- Outside depth FOV: `voxNDC.x/y` outside [0.01, 0.99]

**Step 3: TSDF computation**
```
sDist = depthEyeDist - voxEyeDist       // positive = behind surface
sDist *= saturate(normDot)              // view-aligned correction
sDistNorm = min(sDist / voxelDistance, 1)  // normalized truncation
withinBand = sDistNorm >= -voxelMin / voxelSize
```

**Step 4: Validity checks**
- Depth disparity: raw depth vs dilated depth within `depthDisparityThreshold`
- Surface normal: `normDot > MIN_DOT` (0.3) for occupied voxels
- Exclusion zones: cylinder rejection around tracked heads

**Step 5: Quality computation**
```
distFactor = saturate(1 - voxEyeDist / maxUpdateDist)
angleFactor = saturate(normDot)
quality = distFactor * angleFactor
q2 = quality²    // suppresses low-quality observations quadratically
```

**Step 6: Voxel update** (see Seeding vs Update below)

## 3. Seeding vs Update

### Empty voxel (weight < 0.001) — Seeding path
```
if quality >= MIN_QUALITY_SEED:
    write TSDF = sDistNorm, weight = SEED_WEIGHT
    if camera available and near surface:
        project to camera UV, write initial color
```

### Existing voxel — Update path
```
blend = q2 * blendRate / (1 + weight * stability)
blend = clamp(blend, 0, 0.7)
if blend < 0.005: skip

newTsdf = lerp(oldTsdf, sDistNorm, blend)
newWeight = min(weight + q2 * weightGrowth, maxWeight)
```

### Color update
For existing voxels near the surface (`|sDistNorm| < COLOR_SURFACE_BAND`):
```
colorBlend = quality * 0.3
rgb = oldAlpha < 0.01 ? camColor : lerp(oldRGB, camColor, colorBlend)
newAlpha = min(oldAlpha + quality * 0.05, 1.0)
```

## 4. Pruning

Every `pruneIntervalSeconds` (default 3s), the Prune kernel runs over the entire volume:
```
if 0 < weight < PRUNE_WEIGHT:
    reset voxel to empty (TSDF = -1, weight = 0)
    clear color to (0,0,0,0)
```
Removes barely-observed voxels that were seeded but never confirmed.

## 5. Warmup

After `warmupIntegrations` (default 15) integration frames, the entire volume is cleared. This discards the initial sensor calibration noise that the Quest 3 depth sensor produces during its first ~0.5s of operation.

## 6. GPU Mesh Extraction (Surface Nets)

`MeshExtractor.Extract()` dispatches `GPUSurfaceNets` — the entire pipeline runs on the GPU with zero CPU readback.

### Compute Pipeline (`SurfaceNetsExtract.compute`)

1. **ClearCounters** — Reset vertex and index atomic counters
2. **ClassifyAndEmit** (3D dispatch over volume) — For each voxel cell, check 12 edges for TSDF zero-crossings with confidence gating (`minMeshWeight`). Emit vertex via `InterlockedAdd` compaction into `_Vertices` buffer, store compact index in `_CoordVertMap`.
3. **BuildVertexDispatchArgs** — Set indirect dispatch dimensions from vertex count. All subsequent per-vertex kernels use `DispatchIndirect`.
4. **[optional] SmoothVertices** — HC-Laplacian on GPU (see Stage 2 below)
5. **[optional] PlaneSnap** — Snap to detected planes (see Stage 3)
6. **[optional] TemporalBlend** — Adaptive temporal damping via `RWTexture3D<float4>` (see Stage 4)
7. **GenerateIndices** — Per vertex: check 3 axes for quad emission, write 6 indices per quad via atomic
8. **BuildIndirectArgs** — Pack index count into `DrawProceduralIndirect` args

### Rendering
- `GPUMeshRenderer` calls `Graphics.RenderPrimitivesIndirect` — single draw call replaces per-chunk `MeshRenderer` draws
- `ScanMeshVertexColor.shader` reads from `StructuredBuffer<GPUVertex>` via `SV_VertexID`
- No `Mesh` objects, no `MeshFilter`, no chunk GameObjects

## 6b. GPU Mesh Regularization Pipeline

Three post-processing stages run on the GPU between vertex emit and index generation. Each is independently configurable and can be disabled by setting its iteration/threshold to zero.

```
ClassifyAndEmit → SmoothVertices → PlaneSnap → TemporalBlend → GenerateIndices
                                       ↑
                            PlaneDetector (periodic RANSAC on background thread)
```

### Stage 2: Normal-Aware Vertex Smoothing (`SmoothVertices` kernel)

After ClassifyAndEmit produces raw Surface Nets positions and normals, this kernel smooths vertex positions using a bilateral HC-Laplacian.

- **Adjacency**: 6-connected voxel grid neighbors via `_CoordVertMap` lookup (O(1) per vertex)
- **Bilateral Laplacian**: `L(pᵢ) = Σ(wⱼ · pⱼ) / Σ(wⱼ)` where `wⱼ = max(0, dot(nᵢ, nⱼ))`
- **HC correction** (Vollmer et al.): Prevents volume shrinkage
  1. `q = lerp(pos, L(pos), λ)` — Laplacian step
  2. `result = q − β(q − original)` — pull back toward original position
- **Ping-pong**: Two `GraphicsBuffer` position buffers (`_SmoothPosA`, `_SmoothPosB`) alternate per iteration
- **Default**: 1 iteration, λ = 0.33, β = 0.5

### Stage 3: Plane Detection & Vertex Snapping

#### PlaneDetector (MonoBehaviour, runs periodically)

Sequential RANSAC with axis-aligned bias detects dominant room planes from subsampled mesh vertices:

1. **Vertex subsampling**: Positions/normals are subsampled (strided) from GPU vertex buffer via periodic `AsyncGPUReadback`, capped at `maxSampleVertices` (2048). This bounds RANSAC cost regardless of scene complexity.
2. For up to `maxPlanes` (6) iterations:
   - **RANSAC**: Sample 3 random non-inlier vertices, fit plane via cross product (80 iterations)
   - **Inlier test**: point-to-plane distance < 2cm AND normal alignment > 0.95
   - **Axis bias**: If plane normal is within `axisSnapAngle` (10°) of an axis, snap to that axis
   - **Refinement**: PCA on inliers → recompute plane via smallest eigenvector of covariance matrix
   - Accept plane if inlier count ≥ `minInliers` (30); mark inliers as consumed
   - **Time budget**: Abort early if elapsed time exceeds `timeBudgetMs` (2ms) to prevent frame spikes
3. **Persistence across frames**: Detected planes merge with persistent planes if normal alignment > 0.95 and distance < 5cm. Merged planes blend parameters weighted by inlier count. Unmatched persistent planes decay by `confidenceDecay` per cycle and are removed at confidence 0.
4. **Detection interval**: Runs every 3 mesh cycles (down from 5) since detection is now lightweight.

#### PlaneSnap (`PlaneSnap` compute kernel)

For each vertex (dispatched indirectly), finds the best-matching detected plane:
- Normal alignment: `|dot(vertexNormal, planeNormal)| > 0.9`
- Proximity: `|dot(pos, normal) − d| < planeSnapThreshold`
- Projects vertex onto plane: `pos -= signedDist × normal × confidence`
- Snap strength scales with plane confidence (0.3 initial → 1.0 after many detections)
- Plane data uploaded to GPU as a `StructuredBuffer<PlaneData>` each cycle

### Stage 4: Adaptive Temporal Vertex Damping (`TemporalBlend` kernel)

`RWTexture3D<float4> _TemporalState` stores previous position (xyz) and stability age (w) per voxel, indexed by 3D coordinate. On subsequent extractions:

- **New vertex** (age = −1): Placed instantly at extracted position (α = 1.0, no damping)
- **Deadzone**: If position changed less than `temporalDeadzone` (1mm), old position is kept exactly and age increments
- **Large displacement** (> `convergenceThreshold` = 5mm): α = `alphaMax` (0.85), age resets to 0 — fast convergence
- **Small displacement** (< convergenceThreshold): Age increments, α = `alphaMin + (alphaMax − alphaMin) × exp(−age × decayRate)`
- **Storage**: Uses `RWTexture3D<float4>` instead of structured buffer to stay under Quest 3's 128MB `GraphicsBuffer` limit
- **Default**: αMax = 0.85, αMin = 0.1, decayRate = 0.15, convergenceThreshold = 5mm, deadzone = 1mm

## 7. Camera Projection & Persistent Texturing

Three layers of texturing, in priority order:

### Layer 1: Keyframe Ring Buffer (live, pixel-level, runtime-only)
`KeyframeStore` maintains a ring buffer of N camera frames (default 8) in a `Texture2DArray`:
- **Slot 0**: Always the latest camera frame, updated every integration frame
- **Slots 1–7**: Historical keyframes, inserted when camera moves >0.3m or rotates >20°
- **Eviction**: Oldest historical slot is overwritten when buffer is full
- **Shader**: Fragment shader iterates all keyframes, projects via pinhole model, picks best match by `dot(viewDir, surfaceNormal)`, samples from the Texture2DArray
- **Memory**: 8 × 1280 × 960 × 4 bytes ≈ 40MB
- **NOT persisted**: Keyframes are lost on save/load

### Layer 2: Triplanar World-Space Textures (persistent, ~8mm/texel, saveable)
`TriplanarCache` maintains 3 axis-aligned 2D textures (1024×1024 RGBA8 each):
- **XZ texture**: For Y-dominant normals (floors, ceilings)
- **XY texture**: For Z-dominant normals (front/back walls)
- **YZ texture**: For X-dominant normals (side walls)
- **Sign-aware UV**: Each texture split in half by normal sign (upper = positive, lower = negative) to prevent opposite walls sharing texels
- **Memory**: 3 × 1024 × 1024 × 4 bytes ≈ 12MB
- **Baking**: `TriplanarBake.compute` runs at integration rate, iterating over depth pixels:
  1. Unproject depth pixel to world position
  2. Sample surface normal from depth normals
  3. Project to camera UV, sample camera color with **Reinhard tone mapping** (`color * exposure / (color * exposure + 1)`) to prevent overexposure
  4. Determine dominant triplanar axis from `abs(normal)`
  5. Map to triplanar UV via `SignedTriUV(gsWorldToVoxelUVW(worldPos), normalComponent)`
  6. **Alpha-decaying blend**: `quality * 0.4 * (1 - alpha * 0.8)` — high-confidence texels become nearly immutable (auto-freeze)
- **Shader**: Fragment shader samples all 3 textures using triplanar blending, weighted by `abs(normal)`
- **Persisted**: Save/load as raw RGBA8 data files

### Layer 3: Vertex Colors (fallback, ~5cm/voxel)
Camera colors accumulated into the 3D color volume during TSDF integration. Sampled per-vertex during mesh extraction.

### Fragment Shader Priority Chain
```
_DEBUG_SOLID → _SHOW_NORMALS → _TRIPLANAR_ONLY check →
  Keyframe match (pixel-level) → Triplanar color (~4mm) → Vertex colors (~5cm)
```
The `_TRIPLANAR_ONLY` toggle skips keyframes to evaluate persistence quality in isolation.

### Pinhole Projection Model (shared by all layers)
```
localPos = camInvRot * (worldPos - camPos)
sensorPt = (localPos.xy / localPos.z) * focalLength + principalPoint
scaleFactor = currentRes / sensorRes, normalized
cropMin = sensorRes * (1 - scaleFactor) / 2
uv = (sensorPt - cropMin) / (sensorRes * scaleFactor)
```

## 8. Stability

Temporal blending on the GPU provides implicit convergence-based stability (see Stage 4). Long-stable vertices resist further displacement via exponentially decaying alpha. Triplanar texels also auto-freeze via alpha-decaying blend rate (see Layer 2 above).

## 9. Persistence

`RoomScanPersistence` saves/loads the full scan state to disk.

### Binary Format (`RMSH` v1)
```
Header: magic (RMSH) | version | timestamp
Params: voxelCount (int3) | voxelSize (float) | integrationCount (int) | triplanarRes (int)
TSDF:   length (int) | raw bytes (RG8_SNorm, ~6.2MB)
Color:  length (int) | raw bytes (RGBA8_UNorm, ~12.5MB)
```
Triplanar textures saved separately as 3 raw RGBA8 files.

### Save Pipeline
1. `AsyncGPUReadback` full TSDF volume (slice-by-slice copy)
2. `AsyncGPUReadback` full color volume
3. `TriplanarCache.Save()` writes 3 raw texture files
4. `BinaryWriter` writes header + volume bytes to `persistentDataPath/RoomScans/scan.bin`
5. Triggered by: periodic autosave (every 60s), `OnApplicationPause`, `OnApplicationQuit`, or manual call

### Load Pipeline
1. Read binary, validate magic/version/voxel dimensions
2. Create `Texture3D`, `SetPixelData`, `Graphics.CopyTexture` to upload TSDF and color to GPU
3. `TriplanarCache.Load()` restores triplanar textures
4. `MeshExtractor.Reinitialize()` reinitializes GPU buffers and triggers full re-extraction
5. Resume scanning (new observations refine the loaded mesh)

## 10. Exclusion Zones

Cylindrical exclusion zones around tracked transforms (typically the user's head):
- **Radius:** 0.6m (XZ plane)
- **Top:** 0.25m above head
- **Bottom:** 1.7m below head

Voxels inside any exclusion cylinder are skipped during integration, preventing the user's body from being reconstructed.

## 11. Key Parameters

### Volume
| Parameter | Default | Description |
|-----------|---------|-------------|
| `voxelCount` | 160×128×160 | Volume resolution |
| `voxelSize` | 0.05m | Voxel edge length |
| `voxelDistance` | 0.15m | TSDF truncation distance |
| `voxelMin` | 0.1m | Min distance for integration band |

### Integration
| Parameter | Default | Description |
|-----------|---------|-------------|
| `depthDisparityThreshold` | 0.5m | Max raw-vs-dilated depth difference |
| `maxUpdateDist` | 5.0m | Far plane for integration |
| `minUpdateDist` | 0.5m | Near plane (rejects close noise) |
| `maxFrustumPositions` | 1,000,000 | Cap on frustum grid cells |

### Convergence
| Parameter | Default | Description |
|-----------|---------|-------------|
| `blendRate` | 0.8 | Blend strength per frame |
| `stability` | 2.5 | Weight resistance to blending |
| `weightGrowth` | 0.025 | Weight gain per quality observation |
| `maxWeight` | 0.5 | Cap on voxel confidence weight |

### Seeding & Pruning (compute shader constants)
| Constant | Value | Description |
|----------|-------|-------------|
| `MIN_QUALITY_SEED` | 0.25 | Min quality to seed an empty voxel |
| `SEED_WEIGHT` | 0.10 | Initial weight for seeded voxels |
| `PRUNE_WEIGHT` | 0.05 | Weight below which voxels are pruned |
| `MIN_DOT` | 0.3 | Min view-normal dot product |
| `COLOR_SURFACE_BAND` | 0.5 | TSDF band for color integration |

### GPU Meshing
| Parameter | Default | Description |
|-----------|---------|-------------|
| `minMeshWeight` | 0.08 | Min voxel weight for Surface Nets to consider |
| `gpuVertexBudgetPercent` | 0.05 | Max vertex fraction of total voxels |
| `meshSmoothIterations` | 1 | Post-extraction vertex smoothing passes (0 = off) |
| `meshSmoothLambda` | 0.33 | Laplacian blend strength per iteration |
| `meshSmoothBeta` | 0.5 | HC back-projection strength (prevents shrinkage) |
| `planeSnapThreshold` | 0.03m | Max vertex-to-plane distance for snapping (0 = off) |
| `planeDetectionInterval` | 3 | Mesh cycles between RANSAC plane detection runs |
| `maxSampleVertices` | 2048 | Cap on vertices fed to RANSAC (strided subsampling) |
| `ransacIterations` | 80 | RANSAC hypothesis iterations per plane candidate |
| `maxPlanes` | 6 | Maximum planes to detect (typical room = floor + ceiling + 4 walls) |
| `temporalAlphaMax` | 0.85 | Blend factor for new/moving vertices (fast convergence) |
| `temporalAlphaMin` | 0.1 | Blend factor for long-stable vertices (resists regression) |
| `temporalDecayRate` | 0.15 | How quickly alpha decays from max to min as vertex stabilizes |
| `convergenceThreshold` | 0.005m | Displacement threshold to consider a vertex still converging |
| `temporalDeadzone` | 0.001m | Position changes below this are suppressed entirely |

### Camera
| Parameter | Default | Description |
|-----------|---------|-------------|
| `cameraExposure` | 3.0 | Exposure boost for dim Quest 3 passthrough |
| `warmupIntegrations` | 15 | Frames before volume clear |
| `pruneIntervalSeconds` | 3.0s | Time between prune passes |

### Scan Rates
| Mode | Integration | Mesh Extraction |
|------|-------------|-----------------|
| Passive | 30 Hz | 30 Hz |
| Guided | 30 Hz | 30 Hz |

### Texture Persistence
| Parameter | Default | Description |
|-----------|---------|-------------|
| `textureResolution` | 1024 | Triplanar texture resolution (per plane) |
| `maxKeyframes` | 8 | Ring buffer size (slot 0 = live, rest historical) |
| `moveThreshold` | 0.3m | Min camera displacement for new keyframe |
| `rotateThresholdDeg` | 20° | Min camera rotation for new keyframe |
| `exposure` (KeyframeStore) | 3.0 | Keyframe display exposure boost |

### Memory Budget (Quest 3)
| Component | Memory |
|-----------|--------|
| TSDF volume (160x128x160 RG8) | ~6.5 MB |
| Color volume (160x128x160 RGBA8) | ~13 MB |
| GPU Surface Nets — coord vertex map (int per voxel) | ~12.5 MB |
| GPU Surface Nets — vertex buffer (163K × 32B) | ~5 MB |
| GPU Surface Nets — index buffer (2.9M × 4B) | ~11 MB |
| GPU Surface Nets — smooth ping-pong (2 × 163K × 12B) | ~4 MB |
| GPU Surface Nets — temporal state (160³ × 16B, RWTexture3D RGBA32F) | ~50 MB |
| Triplanar textures (3x 1024x1024 RGBA8) | ~12 MB |
| Keyframe array (8x 1280x960 RGBA8) | ~40 MB |
| **Total GPU** | **~155 MB** |
| **Persistence on disk** | **~31 MB** |

## 12. Gaussian Splat Export Pipeline

Automatic export of camera keyframes and dense point cloud for PC-side Gaussian Splat training.

### KeyframeCollector (Quest side, automatic)
Runs alongside scanning with no user interaction. Saves posed camera frames to `{persistentDataPath}/GSExport/`:
- **Selection**: Motion-gated — translation > 0.3m OR rotation > 20 deg from any saved keyframe
- **Rejection**: Frames with angular velocity > 120 deg/s are discarded (motion blur)
- **Per frame**: JPEG (1280x960, quality 90) + one JSON line in `frames.jsonl` with:
  - Position (px, py, pz), rotation quaternion (qx, qy, qz, qw)
  - Intrinsics (fx, fy, cx, cy), sensor resolution, current resolution
- **I/O**: `AsyncGPUReadback` → JPEG encode → `Task.Run` file write (zero frame stalls)
- **Deduplication**: Multiple pose entries per image ID may occur; the PC pipeline keeps only the last pose per image
- **Typical output**: 100-300 keyframes, 10-30MB total

### PointCloudExporter (Quest side, periodic)
Exports GPU mesh vertices as binary PLY to `GSExport/points3d.ply`:
- Async GPU readback of the `GPUSurfaceNets` vertex buffer
- Parses `GPUVertex` structs: position (float3), normal (float3), packedColor (uint) → RGB
- Writes position, normal, color per vertex in Unity coordinates (left-handed Y-up)
- Runs every 30s automatically
- Provides dense initialization for GS training (10-100x more points than SfM)

### PC Pipeline (`Tools~/gs_pipeline.py`)
Single-command pipeline: `python Tools~/gs_pipeline.py --package com.your.app`

1. **Pull**: `adb pull` GSExport directory from Quest, symlinks `captures/latest`
2. **Convert**: `frames.jsonl` → COLMAP binary format (`cameras.bin`, `images.bin`, `points3D.bin`)
   - Coordinate transform: Unity (left-handed Y-up) → COLMAP (right-handed Y-down) via `diag(1,-1,1)` flip
   - Single PINHOLE camera model from Quest passthrough intrinsics, with principal point crop adjustment
   - Deduplicates frames by image ID, validates image existence
3. **Train**: Auto-detects best backend — msplat (Metal), gsplat (CUDA), or original 3DGS repo
4. **Output**: Trained `splat.ply` ready for Unity import via any Gaussian Splat renderer

### Coordinate Conversion Detail
Unity uses left-handed Y-up; COLMAP uses right-handed Y-down. The conversion:
- **Positions**: Negate Y component
- **Rotations**: Apply `flip @ R_unity @ flip` where `flip = diag(1, -1, 1)` (determinant = -1, changes handedness)
- **Intrinsics**: Adjust principal point (cx, cy) for center crop from sensor resolution to JPEG resolution
