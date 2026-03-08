# QuestRoomScan — Algorithm Reference

## Overview

Real-time 3D room reconstruction on Quest 3 using a TSDF (Truncated Signed Distance Field) volume, Surface Nets mesh extraction, and camera texture projection.

```
DepthCapture (AR depth frames → normals → dilation)
       │
VolumeIntegrator (TSDF integrate → warmup clear → prune)
       ├── ChunkManager → SurfaceNetsMesher (mesh + vertex colors)
       ├── TriplanarCache (bake camera → 3 world-space textures, persistent)
       └── KeyframeStore (ring buffer of camera frames, live quality)
                │
ScanMeshVertexColor.shader (keyframes → triplanar → vertex color fallback)
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

## 6. Mesh Extraction

### Chunk System
- World divided into chunks of `chunkWorldSize` (default 4m³)
- `ChunkManager.UpdateDirtyChunks()` determines which chunks overlap the depth frustum
- Chunks are enqueued for meshing on background worker threads (default 2)
- Each chunk maps to a voxel sub-region via `WorldToVoxel`

### Async GPU Readback
1. Request TSDF volume slice (`R8G8_SNorm`)
2. `CopySliceRG8Job` extracts R channel (TSDF) and G channel (weight) into separate `NativeArray<sbyte>` arrays
3. Request color volume slice (`RGBA8_UNorm`)
4. Copy color data into `NativeArray<Color32>`

### Confidence-Gated Surface Nets
The mesher uses a **confidence gate**: voxels with weight below `minMeshWeight` are treated as empty, preventing phantom surfaces from low-confidence observations.

**VertexJob** (per voxel cell):
1. For each of 12 edges of the cell's 8 corners:
   - Get TSDF values at both endpoints
   - If weight at either endpoint is below `minMeshWeight`, treat as empty (value = 0)
   - Check for zero crossing (`valA < 0 != valB < 0`)
   - Track "bad crossings" (one side is genuinely empty, i.e., `sbyte.MinValue`)
2. Reject if `numCrossings < 3` or all crossings are bad
3. Average crossing positions → vertex position
4. Gradient from finite differences → vertex normal
5. Look up color from color volume (alpha > 10 threshold)

**IndexJob** (per vertex):
- For each axis (X, Y, Z), check if the voxel and its neighbor have opposite TSDF signs
- Form quad from 4 neighboring vertices, split into 2 triangles
- Winding order based on TSDF sign

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

### Layer 2: Triplanar World-Space Textures (persistent, ~4mm/texel, saveable)
`TriplanarCache` maintains 3 axis-aligned 2D textures (2048×2048 RGBA8 each):
- **XZ texture**: For Y-dominant normals (floors, ceilings)
- **XY texture**: For Z-dominant normals (front/back walls)
- **YZ texture**: For X-dominant normals (side walls)
- **Memory**: 3 × 2048 × 2048 × 4 bytes ≈ 48MB
- **Baking**: `TriplanarBake.compute` runs at integration rate, iterating over depth pixels:
  1. Unproject depth pixel to world position
  2. Sample surface normal from depth normals
  3. Project to camera UV, sample camera color
  4. Determine dominant triplanar axis from `abs(normal)`
  5. Map to triplanar UV via `gsWorldToVoxelUVW(worldPos)` (same 0–1 range as volume)
  6. Quality-weighted running average blend into the triplanar texel
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

## 8. Exclusion Zones

Cylindrical exclusion zones around tracked transforms (typically the user's head):
- **Radius:** 0.6m (XZ plane)
- **Top:** 0.25m above head
- **Bottom:** 1.7m below head

Voxels inside any exclusion cylinder are skipped during integration, preventing the user's body from being reconstructed.

## 9. Key Parameters

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

### Meshing
| Parameter | Default | Description |
|-----------|---------|-------------|
| `minMeshWeight` | 0.08 | Min voxel weight for Surface Nets to consider |
| `chunkWorldSize` | 4m³ | Spatial chunk dimensions |
| `overlap` | 0.25m | Extra voxels per chunk edge |
| `numMeshWorkers` | 2 | Background mesher threads |

### Camera
| Parameter | Default | Description |
|-----------|---------|-------------|
| `cameraExposure` | 3.0 | Exposure boost for dim Quest 3 passthrough |
| `warmupIntegrations` | 15 | Frames before volume clear |
| `pruneIntervalSeconds` | 3.0s | Time between prune passes |

### Scan Rates
| Mode | Integration | Mesh Extraction | Texture |
|------|-------------|-----------------|---------|
| Passive | 3 Hz | 1 Hz | 5 Hz |
| Guided | 8 Hz | 3 Hz | 15 Hz |

### Texture Persistence
| Parameter | Default | Description |
|-----------|---------|-------------|
| `textureResolution` | 2048 | Triplanar texture resolution (per plane) |
| `maxKeyframes` | 8 | Ring buffer size (slot 0 = live, rest historical) |
| `moveThreshold` | 0.3m | Min camera displacement for new keyframe |
| `rotateThresholdDeg` | 20° | Min camera rotation for new keyframe |
| `exposure` (KeyframeStore) | 3.0 | Keyframe display exposure boost |

### Memory Budget (Quest 3)
| Component | Memory |
|-----------|--------|
| TSDF volume (160x128x160 RG8) | ~6.5 MB |
| Color volume (160x128x160 RGBA8) | ~13 MB |
| Triplanar textures (3x 2048x2048 RGBA8) | ~48 MB |
| Keyframe array (8x 1280x960 RGBA8) | ~40 MB |
| **Total** | **~108 MB** |
