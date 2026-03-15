# QuestRoomScan

Real-time 3D room reconstruction on Meta Quest 3. Produces a textured mesh from depth + RGB camera data using GPU TSDF volume integration and Surface Nets mesh extraction, with server-based Gaussian Splat training and on-device rendering via [Unity Gaussian Splatting](https://github.com/arghyasur1991/UnityGaussianSplatting).

## Features

- **GPU TSDF Integration** — Depth frames fused into a signed distance field via compute shaders
- **GPU Surface Nets Meshing** — Fully GPU-driven mesh extraction via compute shaders with zero CPU readback, rendered via a single `Graphics.RenderPrimitivesIndirect` draw call
- **Three-Layer Texturing** — Keyframe ring buffer (pixel-level) → triplanar world-space cache (persistent) → vertex colors (fallback)
- **Mesh Persistence** — Save/load full scan state (TSDF + color volumes + triplanar textures) to disk
- **Temporal Stabilization** — Adaptive per-vertex temporal blending on GPU prevents mesh jitter while allowing fast convergence
- **Exclusion Zones** — Cylindrical rejection around tracked heads prevents body reconstruction
- **Gaussian Splat Training & Rendering** — Keyframe capture + point cloud export → PC server training → trained PLY download → on-device UGS rendering
- **VR Debug Menu** — World-space UI Toolkit menu (controller ray interaction, lazy-follow) for live status, render mode toggle, training control, and data management
- **Render Mode Switching** — Toggle between mesh, Gaussian splat, and combined views at runtime via debug menu or controller binding

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
| `org.nesnausk.gaussian-splatting` | [fork](https://github.com/arghyasur1991/UnityGaussianSplatting) | Gaussian splat rendering with runtime PLY loading |

### Android Permissions

- `com.oculus.permission.USE_SCENE` (depth API / spatial data)
- `horizonos.permission.HEADSET_CAMERA` (passthrough camera RGB access)

## Installation

Add to your project's `Packages/manifest.json`:

```json
{
  "dependencies": {
    "com.genesis.roomscan": "https://github.com/arghyasur1991/QuestRoomScan.git",
    "org.nesnausk.gaussian-splatting": "https://github.com/arghyasur1991/UnityGaussianSplatting.git?path=package#feature/runtime-ply-loading"
  }
}
```

Or clone locally and reference as local packages:

```json
{
  "dependencies": {
    "com.genesis.roomscan": "file:../QuestRoomScan",
    "org.nesnausk.gaussian-splatting": "file:/path/to/UnityGaussianSplatting/package"
  }
}
```

## Quick Start

1. Open the setup wizard: **RoomScan > Setup Scene**
2. The wizard checks prerequisites (AR Session, XR Camera Rig, Occlusion Manager) and adds all required components — including `GaussianSplatRenderer` with UGS shaders and the URP render feature
3. Build and deploy to Quest 3
4. The room mesh appears as you look around — surfaces solidify with repeated observations

### Architecture

```
DepthCapture (AR depth frames → normals → dilation)
       │
VolumeIntegrator (TSDF integrate → warmup clear → prune)
       │
MeshExtractor → GPUSurfaceNets (compute: classify → smooth → snap → temporal → index)
       │         └── GPUMeshRenderer (Graphics.RenderPrimitivesIndirect, single draw call)
       │
       ├── TriplanarCache (bake camera → 3 world-space textures)
       └── KeyframeCollector (ring buffer of camera frames → GSExport/)
                │
                ├── PointCloudExporter (GPU mesh → binary PLY)
                │
                └── GSplatServerClient ──► PC Server (upload ZIP → train → download PLY)
                                               │
                                    GSplatManager + GaussianSplatRenderer (UGS)
                                               │
                                    On-device Gaussian Splat rendering
```

See [ALGORITHM.md](ALGORITHM.md) for the full technical reference.

## Gaussian Splat Pipeline

QuestRoomScan captures keyframes and a dense point cloud during scanning, uploads them to a PC training server, and renders the trained Gaussian splats on-device.

### On-Device (automatic capture)

- **KeyframeCollector**: Motion-gated JPEG frames + camera poses saved to `GSExport/`
- **PointCloudExporter**: GPU mesh vertices exported as binary PLY via async readback

### Server Training (via [RoomScan-GaussianSplatServer](https://github.com/arghyasur1991/RoomScan-GaussianSplatServer))

The companion PC server handles the full training pipeline:

```bash
cd gs-server/server && python main.py --port 8420
cd gs-server/web && npm run dev  # Dashboard at http://localhost:5173
```

The flow:
1. **Upload**: Quest sends a ZIP of keyframes + point cloud to the server
2. **Convert**: Server converts Unity poses + intrinsics to COLMAP binary format, computes scene normalization parameters
3. **Train**: Gaussian Splat training via msplat (Metal), gsplat (CUDA), or 3DGS
4. **Denormalize**: Output PLY is transformed back to world coordinates (reverses nerfstudio-style scene normalization)
5. **Download**: Quest downloads the trained PLY
6. **Render**: `GSplatManager` calls `GaussianSplatPlyLoader.LoadFromPlyBytes()` to parse PLY at runtime, convert to UGS internal format, and render via `GaussianSplatRenderer`

### On-Device Rendering (UGS)

Trained splats are rendered using a [fork of Unity Gaussian Splatting](https://github.com/arghyasur1991/UnityGaussianSplatting/tree/feature/runtime-ply-loading) with added runtime PLY loading:

- **`GaussianSplatPlyLoader`**: Parses binary PLY → converts to UGS VeryHigh (Float32) format → creates GPU buffers directly (no Editor asset pipeline needed)
- **`GaussianSplatRenderer.SetRuntimeSplatData()`**: Accepts pre-built GPU buffer data, bypassing TextAsset/ScriptableObject requirements
- **Coordinate conversion**: COLMAP (right-handed Y-down) → Unity (left-handed Y-up)
- **Render mode switching**: Mesh, Splat, or Both — toggled via debug menu or controller binding without releasing GPU resources

### Debug Menu

A world-space UI Toolkit menu accessible via the Quest's Menu button:

- **Live status**: Scan state, render mode, volume stats, training progress
- **Actions**: Start/Stop scanning, Save/Load scan, Start GS training, Switch render mode, Clear data
- **Training status**: Upload/download progress, training state, iteration count
- **Lazy-follow**: Panel follows head gaze with smooth damping

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
| **Processing** | Cloud (1-8 hours after capture) | Real-time textured mesh on-device, GS training on local PC |
| **Output quality** | Photorealistic Gaussian Splats | Textured mesh (real-time) + on-device GS rendering via UGS |
| **Data access** | No raw file export | Full export: PLY point cloud, JPEG keyframes, camera poses |
| **Extensibility** | Closed, no API | MIT open source, every parameter exposed |
| **GS training** | Handled by Meta's cloud | Your hardware, your choice of backend (msplat/gsplat/3DGS) |
| **Offline use** | Requires upload + cloud processing | Works entirely offline (except GS training on PC) |
| **Integration** | Standalone app | Unity package — embed scanning in your own app |

QuestRoomScan is best suited for developers who need to integrate room scanning into their own applications, want full control over the reconstruction pipeline, or need to work with the raw scan data directly.

## Credits & Prior Art

The TSDF volume integration and Surface Nets meshing approach draws inspiration from [anaglyphs/lasertag](https://github.com/anaglyphs/lasertag) by Julian Triveri & Hazel Roeder (MIT), which demonstrated real-time room reconstruction on Quest 3 inside a mixed reality game.

QuestRoomScan builds on that foundation with significant extensions:

| | lasertag | QuestRoomScan |
|-|----------|---------------|
| **Mesh extraction** | CPU marching cubes from GPU volume | Fully GPU-driven Surface Nets via compute shaders — zero CPU readback, single indirect draw call |
| **Texturing** | Geometry only — no camera RGB texturing | Full camera-based texturing at three resolution tiers: keyframe projection (pixel-level), triplanar world-space cache (~8mm/texel), and vertex colors (~5cm) — all sourced from passthrough camera RGB |
| **Persistence** | None — mesh lost on restart | Save/load of TSDF + color volumes + triplanar textures to disk |
| **Mesh quality** | Basic TSDF blending | Quality² modulation, confidence-gated Surface Nets, warmup clearing, pruning, body exclusion zones, GPU temporal stabilization, RANSAC plane detection & snapping |
| **Gaussian Splatting** | — | Full pipeline: on-device capture → PC server training → on-device UGS rendering with render mode switching |
| **VR UI** | — | World-space debug menu with controller ray interaction, live status, and training controls |
| **Packaging** | Embedded in a game | Standalone Unity package with one-click editor setup wizard |

## License

[MIT](LICENSE.md) — see [LICENSE.md](LICENSE.md) for full text and attribution.
