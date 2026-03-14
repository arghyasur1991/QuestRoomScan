# QuestRoomScan

Real-time 3D room reconstruction on Meta Quest 3. Produces a textured mesh from depth + RGB camera data using GPU TSDF volume integration and Surface Nets mesh extraction, with optional export for Gaussian Splat training.

## Features

- **GPU TSDF Integration** — Depth frames fused into a signed distance field via compute shaders
- **GPU Surface Nets Meshing** — Fully GPU-driven mesh extraction via compute shaders with zero CPU readback, rendered via a single `Graphics.RenderPrimitivesIndirect` draw call
- **Three-Layer Texturing** — Keyframe ring buffer (pixel-level) → triplanar world-space cache (persistent) → vertex colors (fallback)
- **Mesh Persistence** — Save/load full scan state (TSDF + color volumes + triplanar textures) to disk
- **Temporal Stabilization** — Adaptive per-vertex temporal blending on GPU prevents mesh jitter while allowing fast convergence
- **Exclusion Zones** — Cylindrical rejection around tracked heads prevents body reconstruction
- **Room Anchoring** — MRUK-based spatial anchoring keeps mesh locked to the physical room across tracking recenters and boundary exits
- **Boundaryless Mode** — Suppresses Quest Guardian boundary for uninterrupted passthrough scanning
- **Gaussian Splat Export** — Automatic keyframe capture + dense point cloud export for PC-side GS training

## Requirements

- **Unity 6** (6000.x)
- **URP** (Universal Render Pipeline)
- **Meta Quest 3** (depth sensor required)

### Dependencies

| Package | Version | Notes |
|---------|---------|-------|
| `com.unity.xr.arfoundation` | 6.1+ | Depth frame access |
| `com.unity.xr.meta-openxr` | 2.1+ | Bridges Meta depth to AR Foundation |
| `com.unity.xr.openxr` | 1.15+ | OpenXR runtime |
| `com.meta.xr.mrutilitykit` | 85+ | Passthrough camera RGB access |
| `com.unity.burst` | 1.8+ | Required by Collections/Mathematics |
| `com.unity.collections` | 2.4+ | NativeArray for plane detection |
| `com.unity.mathematics` | 1.3+ | Math types used throughout |

### Android Permissions

- `com.oculus.permission.USE_SCENE` (depth API / spatial data)
- `horizonos.permission.HEADSET_CAMERA` (passthrough camera RGB access)

## Installation

Add to your project's `Packages/manifest.json`:

```json
{
  "dependencies": {
    "com.genesis.roomscan": "https://github.com/arghyasur1991/QuestRoomScan.git"
  }
}
```

Or clone locally and reference as a local package:

```json
{
  "dependencies": {
    "com.genesis.roomscan": "file:../QuestRoomScan"
  }
}
```

## Quick Start

1. Open the setup wizard: **RoomScan > Setup Scene**
2. The wizard checks prerequisites (AR Session, XR Camera Rig, Occlusion Manager) and adds all required components
3. Build and deploy to Quest 3
4. The room mesh appears as you look around — surfaces solidify with repeated observations

### Architecture

```
RoomAnchorManager (MRUK scene load → volumeToWorld matrix → boundaryless mode)
       │
RoomScanner (waits for RoomReady, then starts pipeline)
       │
DepthCapture (AR depth frames → normals → dilation)
       │
VolumeIntegrator (TSDF integrate → warmup clear → prune)
       │
MeshExtractor → GPUSurfaceNets (compute: classify → smooth → snap → temporal → index)
       │         └── GPUMeshRenderer (Graphics.RenderPrimitivesIndirect, single draw call)
       │
       ├── PlaneDetector (periodic RANSAC on background thread → persistent plane list)
       ├── TriplanarCache (bake camera → 3 world-space textures)
       └── KeyframeStore (ring buffer of camera frames)
                │
ScanMeshVertexColor.shader (SV_VertexID → keyframes → triplanar → vertex color fallback)
```

See [ALGORITHM.md](ALGORITHM.md) for the full technical reference.

## Gaussian Splat Export

QuestRoomScan can automatically capture keyframes and a dense point cloud during scanning for PC-side Gaussian Splat training. This produces photorealistic scene reconstructions.

### On-Device (automatic)

- **KeyframeCollector**: Saves motion-gated JPEG frames + camera poses to `GSExport/`
- **PointCloudExporter**: Periodically exports GPU mesh vertices as a binary PLY point cloud via async readback

### PC Pipeline

A Python script handles the full workflow from Quest to trained Gaussian Splat:

```bash
# Install dependencies
pip install -r Tools~/requirements.txt
pip install "msplat[cli]"   # Apple Silicon (Metal)
# OR: pip install gsplat    # NVIDIA GPU (CUDA)

# Run the full pipeline: pull → convert → train
python Tools~/gs_pipeline.py --package com.your.app

# Or step by step:
python Tools~/gs_pipeline.py --pull --package com.your.app
python Tools~/gs_pipeline.py --convert-only
python Tools~/gs_pipeline.py --train-only --iterations 30000
```

The pipeline:
1. **Pulls** keyframes and point cloud from Quest via `adb`
2. **Converts** Unity poses + intrinsics to COLMAP binary format
3. **Trains** a Gaussian Splat using the best available backend (msplat, gsplat, or 3DGS)

The trained PLY can be imported into Unity with any Gaussian Splat renderer.

### Supported Training Backends

| Backend | Platform | Install |
|---------|----------|---------|
| [msplat](https://github.com/nicknish/msplat) | Apple Silicon (Metal) | `pip install "msplat[cli]"` |
| [gsplat](https://github.com/nerfstudio-project/gsplat) | NVIDIA GPU (CUDA) | `pip install gsplat` |
| [3DGS](https://github.com/graphdeco-inria/gaussian-splatting) | NVIDIA GPU (CUDA) | Clone repo, pass `--gs-repo` |

## Memory Budget (Quest 3)

| Component | Memory |
|-----------|--------|
| TSDF volume (160x128x160, RG8) | ~6.5 MB |
| Color volume (160x128x160, RGBA8) | ~13 MB |
| GPU Surface Nets (coord map, vertices, indices, smoothing, temporal 3D texture) | ~83 MB |
| Triplanar textures (3x 1024x1024, RGBA8) | ~12 MB |
| Keyframe ring buffer (8x 1280x960, RGBA8) | ~40 MB |
| **Total GPU** | **~155 MB** |

## Comparison with Hyperscape

[Meta Horizon Hyperscape](https://www.meta.com/help/quest/1088536553019177/) is Meta's first-party room scanning app for Quest 3. It produces stunning photorealistic Gaussian Splat captures — significantly higher visual quality than what QuestRoomScan currently achieves. If your goal is purely the best-looking scan, Hyperscape is the better choice today.

QuestRoomScan exists for a different reason: it's **open source, fully on-device, and gives you complete control over the pipeline**.

| | Hyperscape | QuestRoomScan |
|-|------------|---------------|
| **Processing** | Cloud (1-8 hours after capture) | Real-time textured mesh on-device, optional Gaussian Splat training on PC |
| **Output quality** | Photorealistic Gaussian Splats | Textured mesh (real-time) + Gaussian Splats via PC pipeline (lower fidelity than Hyperscape currently) |
| **Data access** | No raw file export | Full export: PLY point cloud, JPEG keyframes, camera poses |
| **Extensibility** | Closed, no API | MIT open source, every parameter exposed |
| **GS training** | Handled by Meta's cloud | Your hardware, your choice of backend (msplat/gsplat/3DGS) |
| **Offline use** | Requires upload + cloud processing | Works entirely offline |
| **Integration** | Standalone app | Unity package — embed scanning in your own app |

QuestRoomScan is best suited for developers who need to integrate room scanning into their own applications, want full control over the reconstruction pipeline, or need to work with the raw scan data directly.

## Credits & Prior Art

The TSDF volume integration and Surface Nets meshing approach draws inspiration from [anaglyphs/lasertag](https://github.com/anaglyphs/lasertag) by Julian Triveri & Hazel Roeder (MIT), which demonstrated real-time room reconstruction on Quest 3 inside a mixed reality game.

QuestRoomScan builds on that foundation with significant extensions:

| | lasertag | QuestRoomScan |
|-|----------|---------------|
| **Mesh extraction** | CPU marching cubes from GPU volume | Fully GPU-driven Surface Nets via compute shaders — zero CPU readback, single indirect draw call |
| **Texturing** | Geometry only — no camera RGB texturing | Full camera-based texturing at three resolution tiers: keyframe projection (pixel-level), triplanar world-space cache (~8mm/texel), and vertex colors (~5cm) — all sourced from passthrough camera RGB |
| **Persistence** | None — mesh lost on restart | Save/load of TSDF + color volumes + triplanar textures to disk with MRUK room anchoring |
| **Spatial anchoring** | — | MRUK room anchor keeps mesh locked to physical room across recenters |
| **Mesh quality** | Basic TSDF blending | Quality² modulation, confidence-gated Surface Nets, warmup clearing, pruning, body exclusion zones, GPU temporal stabilization, RANSAC plane detection & snapping |
| **Gaussian Splat export** | — | On-device keyframe + point cloud capture, PC pipeline for COLMAP conversion and GS training |
| **Packaging** | Embedded in a game | Standalone Unity package with one-click editor setup wizard |

## License

[MIT](LICENSE.md) — see [LICENSE.md](LICENSE.md) for full text and attribution.
