# QuestRoomScan — Algorithm Reference

## Overview

Real-time 3D room reconstruction on Quest 3 using a TSDF (Truncated Signed Distance Field) volume and fully GPU-driven Surface Nets mesh extraction.

```
RoomAnchorManager (OVRSpatialAnchor persistence + MRUK fallback → per-artifact relocation)
       │
RoomScanPersistence (package-based multi-scan persistence, anchor.json, manifest.json)
       │
DepthCapture (AR depth frames → normals → dilation, tracking→world conversion)
       │
VolumeIntegrator (TSDF integrate → warmup clear → prune → bake relocation on load)
       │
MeshExtractor → GPUSurfaceNets (compute shader: classify → compact → smooth → snap → temporal → index)
       │         └── GPUMeshRenderer (Graphics.RenderPrimitivesIndirect, single draw call)
       │
       ├── PlaneDetector (periodic RANSAC on background thread → persistent plane list)
       ├── TriplanarCache (bake camera → 3 world-space textures + depth maps,
       │                    forward-splat relocation on load)
       └── KeyframeCollector (motion-gated JPEG keyframes to disk for refinement/splat)
               │
ScanMeshVertexColor.shader (SV_VertexID + StructuredBuffer → triplanar → vertex color → wireframe)
```

## 1. Volume Layout

### TSDF Volume
- **Format:** `R8G8_SNorm` — 2 bytes per voxel
  - **R channel:** Signed distance to surface, normalized by truncation distance. Range [-1, 1]. Negative = inside surface, positive = outside.
  - **G channel:** Confidence weight [0, 1]. Tracks how many quality observations support this voxel's value.
- **Dimensions:** 256 × 256 × 256 voxels (default)
- **Voxel size:** 0.05 m
- **World coverage:** 12.8m × 12.8m × 12.8m, centered at origin
- **Memory:** ~32 MB
- **Empty marker:** R = `sbyte.MinValue` (-128), G = 0

### Color Volume
- **Format:** `R8G8B8A8_UNorm` — 4 bytes per voxel
  - **RGB:** Accumulated camera color (exposure-boosted, quality-weighted running average)
  - **A:** Coverage weight [0, 1]. Tracks color confidence.
- **Dimensions:** Same as TSDF
- **Memory:** ~64 MB

### Coordinate System
- The volume is always axis-aligned at the world origin — there is no `volumeToWorld` matrix.
- Voxel `(0,0,0)` maps to world `(-VoxelCount/2 * voxelSize)`.
- `gsWorldToVoxelFloat(pos)`: `pos / voxelSize + VoxelCount / 2` (shader helper in `VolumeHelpers.hlsl`).
- `gsVoxelToWorld(idx)`: `(idx + 0.5 - VoxelCount / 2) * voxelSize`.
- On load, if the room anchor has moved, a one-shot `BakeRelocation` resamples the volume into the current frame rather than maintaining a persistent transform offset.

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

Two layers of live texturing, in priority order:

### Layer 1: Triplanar World-Space Textures (persistent, ~8mm/texel, saveable)
`TriplanarCache` maintains 3 axis-aligned 2D color textures (4096×4096 RGBA8 each) plus 3 auxiliary depth textures (4096×4096 R8 each):
- **XZ texture**: For Y-dominant normals (floors, ceilings)
- **XY texture**: For Z-dominant normals (front/back walls)
- **YZ texture**: For X-dominant normals (side walls)
- **Depth textures**: Store the "missing axis" value per texel (XZ→Y, XY→Z, YZ→X). Required for exact 3D reconstruction during relocation (see §9c).
- **Sign-aware UV**: Each texture split in half by normal sign (upper = positive, lower = negative) to prevent opposite walls sharing texels
- **Memory**: Color 3 × 4096 × 4096 × 4 bytes ≈ 192MB, Depth 3 × 4096 × 4096 × 1 byte ≈ 48MB
- **Baking**: `TriplanarBake.compute` runs at integration rate, iterating over depth pixels:
  1. Unproject depth pixel to world position
  2. Sample surface normal from depth normals
  3. Project to camera UV, sample camera color with **Reinhard tone mapping** (`color * exposure / (color * exposure + 1)`) to prevent overexposure
  4. Determine dominant triplanar axis from `abs(normal)`
  5. Map to triplanar UV via `SignedTriUV(gsWorldToVoxelUVW(worldPos), normalComponent)`
  6. **Alpha-decaying blend**: `quality * 0.4 * (1 - alpha * 0.8)` — high-confidence texels become nearly immutable (auto-freeze)
  7. Write missing-axis depth alongside color (e.g., `gsTriDepthXZ_RW[tc] = uvw.y` for the XZ face)
- **Shader**: Fragment shader samples all 3 textures using triplanar blending, weighted by `abs(normal)`
- **Persisted**: Save/load as raw RGBA8 color files + R8 depth files

### Layer 2: Vertex Colors (fallback, ~5cm/voxel)
Camera colors accumulated into the 3D color volume during TSDF integration. Sampled per-vertex during mesh extraction.

### Fragment Shader Priority Chain
```
1. Base color selection:
   _RSTriAvailable → Triplanar sample (with vertex color fallback where no data)
   _RSNormalFallback → Normal-as-color (sensor warmup, no camera data yet)
   else → Vertex colors (~5cm)

2. Freeze tint: applies blue overlay on frozen voxels (unless _RSNoFreezeTint)

3. Wireframe: if _RSWireframe, discard interior fragments, render edges
   blending from white at edge midpoints to vertex color at vertices
```
Triplanar textures and vertex colors are the two live texturing layers. `KeyframeCollector` saves JPEG keyframes to disk for post-scan refinement and Gaussian Splat training — they are not used in the live shader.

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

`RoomScanPersistence` manages a **package-based multi-scan persistence system**. Each scan is a self-contained package with all its artifacts.

### Package Layout
```
{persistentDataPath}/RoomScans/
  manifest.json                    # Global index of all packages
  pkg_20260228_143022/
    scan.bin                       # TSDF + color volumes (RMSH v1 binary)
    anchor.json                    # OVRSpatialAnchor UUID + per-artifact matrices
    triplanar/                     # 6 raw files (3 color + 3 depth)
    keyframes/                     # Copied from GSExport/ (images/*.jpg + frames.jsonl)
    splat.ply                      # Auto-saved when GS training completes
    refined_mesh.bin               # Auto-saved when refinement completes
    refined_atlas.raw              # On-device refined atlas (RGBA32)
    hq_atlas.png                   # Server-side HQ refined atlas (PNG)
```

### anchor.json — Per-Artifact Creation Matrices
```json
{
  "anchorUuid": "abc-123-...",
  "baseMatrixAtSave": [16 floats],
  "splatMatrixAtCreate": [16 floats or null],
  "refinedMatrixAtCreate": [16 floats or null],
  "hqMatrixAtCreate": [16 floats or null]
}
```
All matrices are from localizing the **same** OVRSpatialAnchor (same UUID) in different sessions. Each artifact records the anchor's `localToWorldMatrix` at the time the artifact was created/saved. On load, the anchor is localized to get `M_now`, and per-artifact relocation matrices are computed:
- `R_volume = M_now × baseMatrixAtSave^-1` → BakeRelocation on TSDF + triplanar
- `R_splat = M_now × splatMatrixAtCreate^-1` → SplatHolder transform
- `R_refined = M_now × refinedMatrixAtCreate^-1` → refined mesh vertex MultiplyPoint3x4 + normal MultiplyVector

This allows artifacts created in different sessions (scan in session 1, splat in session 2, refine in session 3) to be correctly relocated when loaded in session 4.

### Binary Format (`RMSH` v1, unchanged)
```
Header:  magic (RMSH) | version (1) | timestamp (int64)
Params:  voxelCount (int3) | voxelSize (float) | integrationCount (int) | triplanarRes (int)
Anchor:  anchorAtSave (Matrix4x4, 16 floats — kept for backward compat)
TSDF:    length (int) | raw bytes (RG8_SNorm)
Color:   length (int) | raw bytes (RGBA8_UNorm)
```
Triplanar data saved separately as 6 raw files: `tri_xz.raw`, `tri_xy.raw`, `tri_yz.raw` (RGBA8 color) + `depth_xz.raw`, `depth_xy.raw`, `depth_yz.raw` (R8 depth).

### Save Pipeline (`SaveToNewPackageAsync`)
1. Generate `pkg_{timestamp}` directory
2. `AsyncGPUReadback` full TSDF + color volumes (slice-by-slice)
3. `RoomAnchorManager.CreateAndSaveSpatialAnchorAsync()` at MRUK floor position → persisted `OVRSpatialAnchor`
4. Write `scan.bin` (same v1 format) with anchor matrix
5. Write `anchor.json` with UUID + `baseMatrixAtSave`
6. Save triplanar textures + depth maps
7. Copy `GSExport/` contents to `keyframes/`
8. Update `manifest.json`
9. Set as `ActivePackageId` — subsequent artifact saves go here automatically

### Artifact Auto-Save (`SaveArtifactAsync`)
Called automatically when: splat download completes, on-device refinement finishes, HQ refinement finishes.
1. Write artifact file(s) to active package directory
2. Record current anchor localization matrix as `*MatrixAtCreate` in `anchor.json`
3. Update manifest flags

### Load Pipeline (`LoadPackageAsync`)
1. Read `scan.bin` + `anchor.json` in background
2. Validate voxel grid compatibility
3. Upload TSDF + color to GPU
4. Localize spatial anchor: `RoomAnchorManager.LoadSpatialAnchorAsync(uuid)` → `M_now`
5. If spatial anchor fails, fall back to MRUK floor anchor (with stabilization polling)
6. Compute per-artifact relocation matrices from `anchor.json`
7. `BakeRelocation(R_volume)` on TSDF volume + triplanar
8. Load splat with `R_splat` applied to `SplatHolder` transform
9. Load refined mesh with `R_refined` applied to vertex positions and normals
10. Rebuild GPU mesh from relocated volume

### 9b. TSDF Volume Bake Relocation (`BakeRelocation` kernel)

Resamples the entire TSDF + color volume from the old coordinate frame into the current identity frame using trilinear interpolation:

```
BakeRelocation(uint3 id):
  dstLocal = (id + 0.5 - VoxelCount/2) × voxelSize
  srcLocal = invRelocation × dstLocal
  uvw = srcLocal / voxelSize + VoxelCount/2 → normalized [0,1]
  if out-of-bounds: write empty voxel
  else: SampleLevel(srcTsdf, uvw, 0), SampleLevel(srcColor, uvw, 0)
```

- **Dispatch**: `[numthreads(4,4,4)]` over full volume grid
- **Swap strategy**: Writes to fresh `RenderTexture` pair, then swaps and destroys originals — avoids `Graphics.CopyTexture` on 3D RTs which can silently fail on Vulkan/Quest
- **Rebinds**: After swap, all compute kernel UAVs and global shader textures are rebound to the new volumes

### 9c. Triplanar Forward-Splat Relocation

Triplanar textures encode a 2D projection of 3D surface data — the axis aligned with the dominant normal is discarded. Inverting this projection requires the "missing axis" value, which is stored per texel during scanning as an auxiliary R8 depth texture (one per face: XZ→Y, XY→Z, YZ→X).

#### Depth Storage (scan-time)
The `BakeTriplanar` compute kernel writes the missing axis value alongside color:
```
if face == XZ: gsTriDepthXZ_RW[tc] = uvw.y
if face == XY: gsTriDepthXY_RW[tc] = uvw.z
if face == YZ: gsTriDepthYZ_RW[tc] = uvw.x
```

#### Forward Splat (load-time, `ForwardSplatRelocation` kernel)
For each old texel with stored depth, reconstructs the exact 3D position, applies the relocation matrix, and scatter-writes to the correct new triplanar face:

```
ForwardSplatRelocation(uint2 id):
  oldColor = srcTri[id]           // skip if alpha < 0.01
  depth = srcDepth[id]            // stored missing axis [0,1]
  uv = (id + 0.5) / triSize      // decode sign-split half

  // Reconstruct 3D from (2D UV + stored depth)
  if srcFace == XZ: oldUVW = (u, depth, v),  faceN = (0, sign, 0)
  if srcFace == XY: oldUVW = (u, v, depth),  faceN = (0, 0, sign)
  if srcFace == YZ: oldUVW = (depth, u, v),  faceN = (sign, 0, 0)

  oldWorldPos = (oldUVW × voxCount - voxCount/2) × voxSize
  newWorldPos = R × oldWorldPos
  newN = normalize(R_3x3 × faceN)

  // Determine new triplanar face and UV
  newUVW = saturate(newWorldPos / voxSize + voxCount/2) / voxCount
  newFace = dominant axis of |newN|
  newTriUV = SignedTriUV(newUVW, newN)

  // Scatter-write to destination UAV
  dstFace[newTriUV] = oldColor
```

- **Dispatch**: 3 passes (one per source face), each writing to all 3 destination face UAVs simultaneously
- **Source textures**: Loaded as `Texture2D` from raw bytes (not `Graphics.Blit` → RT), bypassing Vulkan SRV layout transition issues on Quest
- **Coverage**: Followed by 4 compute dilation passes (`DilateTriplanar` kernel, 3×3 average of non-empty neighbors, ping-pong between two RT sets) to fill sparse gaps from scatter-write quantization
- **Depth RTs after relocation**: Fresh empty R8 `RenderTexture`s are created for subsequent live scanning

## 10. Room Anchoring

`RoomAnchorManager` provides two anchoring mechanisms:

### OVRSpatialAnchor (Primary — Persistence)
The primary cross-session relocation mechanism uses Meta's `OVRSpatialAnchor` API:
- **Create**: At save time, a spatial anchor is created at the MRUK floor position, persisted to device storage via `SaveAnchorAsync()`. The anchor's UUID and `localToWorldMatrix` are stored in `anchor.json`.
- **Load**: At load time, the anchor is loaded via `LoadUnboundAnchorsAsync()` and localized via `LocalizeAsync()`. The localized anchor's `localToWorldMatrix` provides `M_now` for computing relocation.
- **Erase**: When a package is deleted, the spatial anchor is erased from persistent storage via `EraseAnchorsAsync()`.
- **Transform stabilization**: After creation or localization, the anchor transform is polled for 5 consecutive frames with < 1mm movement before reading the matrix (prevents jitter from ongoing SLAM refinement).

### MRUK Floor Anchor (Fallback — Runtime)
On startup, `RoomAnchorManager` loads the device's scene model via `MRUK.LoadSceneFromDevice()` and grabs the first floor anchor's `localToWorldMatrix`. This serves two purposes:
1. **Runtime world-locking**: MRUK world-locks the scene mesh to the room, providing stable camera-to-world conversion
2. **Fallback**: If spatial anchor localization fails (e.g., device storage cleared, major room changes), the MRUK floor anchor is used for relocation instead

### Per-Artifact Relocation
Artifacts created in different sessions (scan in session 1, splat in session 2, refined mesh in session 3) each record the spatial anchor's `localToWorldMatrix` at creation time. On load in session 4:
```
R_volume  = M_now × baseMatrixAtSave^-1
R_splat   = M_now × splatMatrixAtCreate^-1
R_refined = M_now × refinedMatrixAtCreate^-1
```
If an artifact's creation matrix is null (legacy data), it falls back to `baseMatrixAtSave`. Raw artifact files are never rewritten — each stays in the coordinate space it was created in.

### Tracking-Space to World-Space Conversion
MRUK's world-lock can reposition the `TrackingSpace` transform each frame. `DepthCapture.TrackingToWorld(Pose)` converts camera poses from tracking space to world space before passing them to `VolumeIntegrator`, `TriplanarCache`, and `KeyframeCollector`.

### Startup Sequence
`RoomScanner.Start()` waits for `RoomAnchorManager.RoomReady` before beginning scanning or loading saved data.

## 11. Exclusion Zones

Cylindrical exclusion zones around tracked transforms (typically the user's head):
- **Radius:** 0.6m (XZ plane)
- **Top:** 0.25m above head
- **Bottom:** 1.7m below head

Voxels inside any exclusion cylinder are skipped during integration, preventing the user's body from being reconstructed.

## 12. Key Parameters

### Volume
| Parameter | Default | Description |
|-----------|---------|-------------|
| `voxelCount` | 256×256×256 | Volume resolution |
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
| Active | 30 Hz | 30 Hz |

### Texture Persistence
| Parameter | Default | Description |
|-----------|---------|-------------|
| `textureResolution` | 4096 | Triplanar texture resolution (per plane, applies to both color and depth) |
| `moveThreshold` | 0.3m | Min camera displacement for new keyframe (KeyframeCollector) |
| `rotateThresholdDeg` | 20° | Min camera rotation for new keyframe (KeyframeCollector) |

### Room Anchoring & Relocation
| Parameter | Default | Description |
|-----------|---------|-------------|
| `RoomAnchorManager` | enabled | Disable to skip MRUK + spatial anchor (identity volume placement) |
| Relocation dilation passes | 4 | Compute dilation passes after forward-splat (fills coverage gaps) |
| Spatial anchor creation timeout | 5s | Max wait for `OVRSpatialAnchor.Created` |
| Spatial anchor localize timeout | 10s | Max wait for `LocalizeAsync` completion |
| Transform stabilization | 5 frames, < 1mm | Stable consecutive frames before reading anchor matrix |

### Memory Budget (Quest 3)
| Component | Memory |
|-----------|--------|
| TSDF volume (256³ RG8_SNorm) | ~32 MB |
| Color volume (256³ RGBA8_UNorm) | ~64 MB |
| GPU Surface Nets — coord vertex map (256³ × 4B) | ~64 MB |
| GPU Surface Nets — vertex buffer (838K × 32B at 5% budget) | ~26 MB |
| GPU Surface Nets — index buffer (838K × 18 × 4B) | ~58 MB |
| GPU Surface Nets — smooth ping-pong (2 × 838K × 12B) | ~19 MB |
| GPU Surface Nets — temporal state (256³ × 16B, RWTexture3D RGBA32F) | ~256 MB |
| Triplanar color textures (3 × 4096² RGBA8) | ~192 MB |
| Triplanar depth textures (3 × 4096² R8) | ~48 MB |
| **Total GPU (with triplanar)** | **~759 MB** |
| **Total GPU (without triplanar)** | **~519 MB** |
| **Persistence on disk** | **~34 MB** |

## 13. Gaussian Splat Pipeline

End-to-end pipeline: on-device keyframe + point cloud capture → server-based COLMAP conversion + GS training → on-device rendering via Unity Gaussian Splatting (UGS).

### 13.1 KeyframeCollector (Quest, automatic)
Runs alongside scanning with no user interaction. Saves posed camera frames to `{persistentDataPath}/GSExport/`:
- **Selection**: Motion-gated — translation > 0.3m OR rotation > 20 deg from any saved keyframe
- **Rejection**: Frames with angular velocity > 120 deg/s are discarded (motion blur)
- **Per frame**: JPEG (1280x960, quality 90) + one JSON line in `frames.jsonl` with:
  - Position (px, py, pz), rotation quaternion (qx, qy, qz, qw)
  - Intrinsics (fx, fy, cx, cy), sensor resolution, current resolution
- **I/O**: `AsyncGPUReadback` → JPEG encode → `Task.Run` file write (zero frame stalls)
- **Deduplication**: Multiple pose entries per image ID may occur; the server keeps only the last pose per image
- **Typical output**: 100-300 keyframes, 10-30MB total

### 13.2 PointCloudExporter (Quest, periodic)
Exports GPU mesh vertices as binary PLY to `GSExport/points3d.ply`:
- Async GPU readback of the `GPUSurfaceNets` vertex buffer
- Parses `GPUVertex` structs: position (float3), normal (float3), packedColor (uint) → RGB
- Writes position, normal, color per vertex in Unity coordinates (left-handed Y-up)
- Runs every 30s automatically
- Provides dense initialization for GS training (10-100x more points than SfM)

### 13.3 Server Training (RoomScan-GaussianSplatServer)

The Quest app's `GSplatServerClient` uploads a ZIP of keyframes + point cloud to a PC-based FastAPI server (`/upload?iterations=N`), then polls for status and downloads the result. Training iterations are configurable via the inspector (`trainingIterations`, default 7000).

#### COLMAP Conversion
`frames.jsonl` → COLMAP binary format (`cameras.bin`, `images.bin`, `points3D.bin`):
- Coordinate transform: Unity (left-handed Y-up) → COLMAP (right-handed Y-down) via `diag(1,-1,1)` flip
- Single PINHOLE camera model from Quest passthrough intrinsics, with principal point crop adjustment
- Deduplicates frames by image ID, validates image existence

#### Scene Normalization
Computed during COLMAP conversion and saved to `scene_norm.json`:
- **Center** = mean of camera positions in COLMAP space
- **Scale** = `1 / mean(distance_from_center)`
- Required because training backends (msplat/nerfstudio) internally normalize the scene but don't expose the parameters

#### Training
Auto-detects best backend — msplat (Metal), gsplat (CUDA), or original 3DGS repo. Default 7,000 iterations (configurable via `GSplatServerClient.trainingIterations`).

#### Denormalization
After training, `denormalize_ply()` reverses the scene normalization on the output `splat.ply`:
- **Positions**: `P_world = P_normalized × avg_dist + center`
- **Scales** (log-space): `s_world = s_normalized + ln(avg_dist)`

The output PLY is in COLMAP world coordinates (right-handed Y-down).

#### Run Management
Each upload creates a timestamped directory. A `current_run` symlink points to the active run. Past runs can be browsed, activated, or deleted via the web dashboard API.

### 13.4 On-Device Rendering (UGS)

Trained PLY is rendered using a [fork of Unity Gaussian Splatting](https://github.com/arghyasur1991/UnityGaussianSplatting) with runtime loading support.

#### Runtime PLY Loading (`GaussianSplatPlyLoader`)
Parses binary little-endian PLY and converts to UGS internal format (VeryHigh / Float32):

1. **PLY header parsing** from `byte[]` — extracts vertex count, stride, attribute types
2. **Attribute mapping**: Raw PLY fields → `InputSplatData` struct (position, normals, DC color, 15 SH bands × 3 channels, opacity, scale, rotation)
3. **SH reordering**: PLY stores coefficients per-channel (all R, then all G, then all B); UGS expects coeff-major order (R0 G0 B0, R1 G1 B1, ...) — `ReorderSHs()` transposes in-place
4. **Linearization** (`LinearizeDataJob`, parallel):
   - Rotation: normalize → swizzle → PackSmallest3 → Norm10 packed uint
   - Scale: `exp(log_scale)` → linear
   - Color: `SH0ToColor(dc0)` → sigmoid → linear RGB
   - Opacity: `sigmoid(raw_opacity)`
5. **COLMAP → Unity conversion** (`ConvertColmapJob`): negate Y position
6. **Buffer construction**:
   - Position: 3 × Float32 per splat
   - Other: Norm10-packed rotation (uint32) + 3 × Float32 scale
   - Color: Float32×4 (RGBA) in Morton-ordered texture layout (`SplatIndexToTextureIndex` via `DecodeMorton2D_16x16`)
   - SH: Float32 table (15 bands × 3 channels per splat)
7. **Bounds**: AABB computed from all splat positions

#### `GaussianSplatRenderer.SetRuntimeSplatData()`
Accepts pre-built `NativeArray<byte>` buffers and directly creates GPU resources:
- `GraphicsBuffer` for positions, other (rot+scale), and SH data
- `Texture2D` for Morton-ordered color data
- Stores format metadata (`VectorFormat.Float32`, `SHFormat.Float32`, `ColorFormat.Float32x4`)
- No dependency on `GaussianSplatAsset`, `TextAsset`, or `ScriptableObject`

#### Quest 3 Stereo Rendering Optimizations
The UGS fork includes several Quest 3-specific optimizations:
- **Per-eye stereo matrices**: Explicit per-eye model-view and projection matrices passed to the compute shader for correct covariance projection in VR (avoids artifacts from using mono matrices)
- **Shared covariance/SH**: Covariance, SH evaluation, and color are computed once for the left eye and reused for the right eye — only the clip position is recomputed
- **Max splat count**: Configurable limit (`m_MaxSplatCount`) to cap rendered splats after sorting, useful for mobile perf tuning
- **Fragment shader**: Uses `clip()` instead of `discard` for better Adreno TBDR performance

#### Visibility Control
`GaussianSplatRenderer.renderVisible` boolean — checked in `GatherSplatsForCamera()` — allows toggling rendering without disabling the component or releasing GPU resources. Used by `GSplatManager.RenderVisible` → `RoomScanner.ApplyRenderMode()` for render mode switching. The `ScanRenderMode` enum controls which representation is active: `Wireframe`, `Vertex`, `Triplanar`, `Refined`, `HQRefined`, `Splat`, or `None`. `CycleRenderMode()` skips modes whose backing data or module is not present (e.g., Triplanar requires `TriplanarCache`, Splat requires trained data).

### 13.5 Coordinate Conversion Detail
Unity uses left-handed Y-up; COLMAP uses right-handed Y-down. The full round-trip:

**Quest → Server (export)**:
- **Positions**: Negate Y component
- **Rotations**: Apply `flip @ R_unity @ flip` where `flip = diag(1, -1, 1)` (determinant = -1, changes handedness)
- **Intrinsics**: Adjust principal point (cx, cy) for center crop from sensor resolution to JPEG resolution

**Server → Quest (denormalized PLY)**:
- PLY is in COLMAP world coordinates (Y-down)
- `GaussianSplatPlyLoader` negates Y position during loading to convert back to Unity space

## 14. Texture Refinement Pipeline

Post-processing pipeline that produces a sharp UV-mapped texture atlas from saved keyframes, replacing the blurry triplanar vertex-color texturing. Uses the same keyframes collected for Gaussian Splat training (§13.1).

### 14.1 Mesh Simplification (optional, meshoptimizer — currently disabled)

Before UV unwrapping, the GPU Surface Nets mesh can be optionally decimated using [meshoptimizer](https://github.com/zeux/meshoptimizer) v1.0 (`meshopt_simplify`):

- **Input**: Original mesh positions + index buffer from GPU readback
- **Operation**: Quadric error metric simplification — removes triangles while preserving mesh topology and surface shape. Only the index buffer changes; vertex positions are untouched.
- **Target**: Configurable ratio, inspector slider `decimationRatio ∈ [0.1, 1.0]`.
- **Performance**: <5ms even for 100k triangles on ARM64

> **Status: Disabled by default** (`decimationRatio = 1.0`). Mesh decimation degrades atlas baking quality and performance — the simplified index buffer produces poor UV charts and misaligned projections during atlas baking. Until the interaction between meshoptimizer simplification and the atlas bake pipeline is resolved, decimation should remain off.

### 14.2 UV Unwrapping (xatlas)

[xatlas](https://github.com/jpcy/xatlas) generates a UV atlas via native C++ P/Invoke:

```
Create atlas → AddMesh(positions, normals, indices) → Generate(chartOpts, packOpts) → read output
```

**Charting phase** segments the mesh into charts (connected patches with low angular distortion):
- `maxCost` (default 1.5, xatlas default 2.0): Chart growth cost threshold. Lower = more, smaller charts = faster parameterization at the cost of more seams.
- `normalDeviationWeight`, `straightnessWeight`, `normalSeamWeight`: Control chart boundary placement relative to surface curvature and existing seams.
- `maxIterations` = 1: Single pass (minimum).

**Packing phase** arranges charts into the atlas texture:
- `resolution` (default 2048): Atlas resolution in pixels.
- `blockAlign` (default true, xatlas default false): Aligns charts to 4×4 pixel blocks. Dramatically reduces packing search space at slight atlas utilization cost.
- `padding` = 2: Pixels between charts to prevent bleeding during bilinear filtering.
- `bilinear` = true: Reserves extra space for bilinear filter sampling.

**Output**: New vertex buffer (split at UV seams), UV coordinates in atlas-pixel space, index buffer, and xref array mapping each output vertex back to the original mesh vertex.

All xatlas options are exposed through a flat C API (`xatlas_generate_opts`) and configurable from the Inspector via `UnwrapOptions`.

### 14.3 GPU Atlas Baking (Compute Shader)

`AtlasBakeCompute.compute` processes each keyframe to project camera imagery onto the UV atlas. All data is in `StructuredBuffer`s with integer pixel indexing — no render targets, no Y-axis ambiguity.

**Three kernels, dispatched per keyframe:**

1. **ClearDepth** (`[numthreads(256,1,1)]`): Fills depth buffer with 999 (far sentinel). One dispatch per keyframe to reset occlusion.

2. **BuildDepth** (`[numthreads(64,1,1)]`): Per original-mesh triangle, rasterizes in screen space using the keyframe's view/projection. Writes `InterlockedMin(asuint(z))` to build an occlusion depth map. This uses the *original* (pre-decimation) mesh to ensure accurate occlusion.

3. **BakeAtlas** (`[numthreads(64,1,1)]`): Per UV-unwrapped triangle:
   - Rasterizes bounding box in atlas UV space
   - Barycentric test for point-in-triangle
   - Interpolates 3D world position from barycentrics
   - Projects to keyframe screen space via intrinsics (fx, fy, cx, cy with crop offset)
   - **Bounds check**: Discards if outside image
   - **Occlusion check**: Compares projected depth against depth buffer (with 0.05 tolerance)
   - **Score**: `dot(surfaceNormal, viewDirection)` — prefers head-on views
   - **Atomic best-score selection**: `InterlockedMax(_ScoreBuf[texelIdx], asuint(score))` — since scores are positive floats, `asuint()` preserves ordering. Color is written only when the thread wins the comparison.

**Keyframe processing** is sequential from C#: decode JPEG → `GetPixels32()` → upload to `ComputeBuffer` → dispatch 3 kernels → `await Task.Yield()`. Score and atlas buffers persist across keyframes (best score accumulates).

**Post-processing** (CPU):
- **Dilation**: Fills empty texels at UV island edges by averaging non-empty neighbors (multiple passes)
- **Denoise** (optional, `skipDenoise` toggle): Median-like filter to remove speckle noise from misaligned projections. GPU compute bake produces fewer speckles than CPU bake, so this is off by default.

**CPU fallback**: `BakeAtlasCPUAsync` implements identical logic in C# with `unsafe` pointer access. Used when compute shader is null (`forceCpuBake` toggle).

### 14.4 Render Modes

`ScanRenderMode` controls which representation of the scan is displayed. `CycleRenderMode()` steps through in the order below, automatically skipping modes whose backing data or module is not present.

| Mode | Renderer | Availability | Description |
|------|----------|-------------|-------------|
| **Wireframe** | GPU mesh (`ScanMeshVertexColor.shader`) | Always | Transparent mesh with white edges blending to vertex colors at vertices. Barycentric edge detection with configurable thickness (`_RSWireThickness`). Interior fragments discarded. |
| **Vertex** | GPU mesh (`ScanMeshVertexColor.shader`) | Always (default) | Live GPU mesh with vertex colors only (~5cm resolution). Triplanar texturing forced off. |
| **Triplanar** | GPU mesh (`ScanMeshVertexColor.shader`) | `TriplanarCache` attached | Live GPU mesh with triplanar-projected camera textures (~8mm/texel) with vertex color fallback where data is missing. |
| **Refined** | `Mesh` object + `RefinedMesh.shader` | After on-device refinement | UV-unwrapped mesh with on-device baked atlas. Standard unlit UV-mapped rendering. |
| **HQRefined** | `Mesh` object + `RefinedMesh.shader` | After server HQ refinement | Same mesh as Refined, with server-enhanced atlas (Real-ESRGAN + LaMa). |
| **Splat** | `GaussianSplatRenderer` (UGS) | After GS training completes | Gaussian Splat point cloud rendered from server-trained PLY data. |
| **None** | — | Always | All scan rendering disabled (GPU mesh hidden, splat hidden, refined hidden). |

The **Freeze Tint** toggle (`ShowFreezeTint`) is independent of render mode — it applies a blue overlay on frozen voxels in Vertex, Triplanar, and Wireframe modes via the `_RSNoFreezeTint` shader global.

Both refined atlases and mesh data persist to disk via `RoomScanPersistence` and survive app restarts / scan reloads.

### 14.5 Server-Side HQ Refinement

Server-side atlas enhancement via Real-ESRGAN super-resolution + LaMa inpainting:
1. Client uploads the on-device refined atlas as PNG to `{serverUrl}/refine-texture/upload`
2. Server applies Real-ESRGAN (2x or 4x, configurable via `hqRefineScale`) for super-resolution
3. Server applies LaMa inpainting to fill gaps in the upscaled atlas
4. Client downloads the enhanced atlas (`hq_atlas.png`)

> **Note:** An earlier differentiable-rendering-based HQ path (PyTorch gradient-based texture optimization) is non-functional and not exposed. The Real-ESRGAN + LaMa pipeline described above is the working path.
