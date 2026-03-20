# QuestRoomScan

Real-time 3D room reconstruction on Meta Quest 3. Produces a textured mesh from depth + RGB camera data using GPU TSDF volume integration and Surface Nets mesh extraction, with server-based Gaussian Splat training and on-device rendering via [Unity Gaussian Splatting](https://github.com/arghyasur1991/UnityGaussianSplatting).

## Features

- **GPU TSDF Integration** — Depth frames fused into a signed distance field via compute shaders
- **GPU Surface Nets Meshing** — Fully GPU-driven mesh extraction via compute shaders with zero CPU readback, rendered via a single `Graphics.RenderPrimitivesIndirect` draw call
- **Three-Layer Texturing** — Triplanar world-space cache (persistent ~8mm/texel) → vertex colors (~5cm fallback) — all sourced from passthrough camera RGB. Keyframes captured as motion-gated JPEGs to disk for Gaussian Splat training.
- **Mesh Persistence** — Save/load full scan state (TSDF + color volumes + triplanar textures + depth maps + room-anchor pose) to disk (`scan.bin` format v1), auto-save on quit
- **MRUK Room Anchoring** — `RoomAnchorManager` uses Meta's MRUK floor anchor to compute a relocation matrix across sessions. On load, TSDF volume and triplanar textures are baked into the current coordinate frame via compute-shader relocation, keeping the scan aligned with the physical room
- **Temporal Stabilization** — Adaptive per-vertex temporal blending on GPU prevents mesh jitter while allowing fast convergence
- **Exclusion Zones** — Cylindrical rejection around tracked heads prevents body reconstruction (configurable radius and height, up to 64 zones)
- **Gaussian Splat Training & Rendering** — Keyframe capture + point cloud export → PC server training → trained PLY download → on-device UGS rendering
- **VR Debug Menu** — World-space UI Toolkit panel with controller ray interaction, lazy-follow gaze tracking, live status, server training controls, persistence management, and FPS display
- **Render Mode Switching** — Cycle between Mesh, Splat, and Both views at runtime via debug menu or controller binding (default: A/X button)

## Requirements

- **Unity 6** (6000.x)
- **URP** (Universal Render Pipeline)
- **Meta Quest 3** (depth sensor required)

### Dependencies

| Package | Version | Notes |
|---------|---------|-------|
| `com.unity.xr.arfoundation` | 6.1+ | Depth frame access |
| `com.unity.render-pipelines.universal` | 17.0+ | URP rendering pipeline |
| `com.meta.xr.mrutilitykit` | 85+ | Passthrough camera RGB access |
| `com.unity.burst` | 1.8+ | Required by Collections/Mathematics |
| `com.unity.collections` | 2.4+ | NativeArray for plane detection |
| `com.unity.mathematics` | 1.3+ | Math types used throughout |
| `org.nesnausk.gaussian-splatting` | [fork](https://github.com/arghyasur1991/UnityGaussianSplatting) | Gaussian splat rendering with runtime PLY loading |

Additional project-level dependencies (not in `package.json` — installed via Meta's SDK or XR plugin management):
- `com.unity.xr.meta-openxr` (bridges Meta depth to AR Foundation)
- `com.unity.xr.openxr` (OpenXR runtime)
- `com.meta.xr.sdk.core` (OVRInput, OVRCameraRig)

### Android Permissions

- `com.oculus.permission.USE_SCENE` (depth API / spatial data)
- `horizonos.permission.HEADSET_CAMERA` (passthrough camera RGB access)

## Installation

Add to your project's `Packages/manifest.json`:

```json
{
  "dependencies": {
    "com.genesis.roomscan": "https://github.com/arghyasur1991/QuestRoomScan.git",
    "org.nesnausk.gaussian-splatting": "https://github.com/arghyasur1991/UnityGaussianSplatting.git?path=package#main"
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

1. Create a new blank URP scene
2. Add a **Camera Rig** and **Passthrough** from Meta's Building Blocks (`Menu > Meta > Building Blocks`). The Camera Rig provides `OVRCameraRig` and the Passthrough block enables the passthrough layer — both are required before running the wizard.
3. Open the setup wizard: **RoomScan > Setup Scene**
4. The wizard checks prerequisites (AR Session, AROcclusionManager), configures project settings (boundaryless manifest, cleartext HTTP for LAN server), and adds all required components — including `GaussianSplatRenderer` with UGS shaders, the URP render feature, VR input handlers, and debug menu
5. Build and deploy to Quest 3
6. The room mesh appears as you look around — surfaces solidify with repeated observations

## Usage Flow

### Scanning

Scanning starts automatically on launch (configurable via `autoStartOnLoad`). As you look around:

1. **Depth integration**: Each depth frame is fused into the TSDF volume with color from the passthrough camera
2. **Mesh extraction**: GPU Surface Nets extracts a mesh from the volume every few frames (after a minimum number of integrations)
3. **Texturing**: Camera RGB is baked into triplanar world-space textures for persistent surface color
4. **Keyframe capture**: Motion-gated JPEG snapshots + camera poses are saved to `GSExport/` on disk — these are used later for Gaussian Splat training
5. **Point cloud export**: GPU mesh vertices are auto-exported as `points3d.ply` every 30 seconds (configurable)

**Tips for a good scan**: Move slowly around the room. Look at surfaces from multiple angles — repeated observations from different viewpoints improve mesh quality. Make sure to cover walls, floor, ceiling, and furniture from several directions before training.

### Freeze / Unfreeze

When a region of the mesh looks good and you don't want further integration to degrade it:

- **Freeze In View** (Y/B button): Locks all voxels currently in your camera frustum. Frozen voxels are skipped during integration — their geometry and color are preserved exactly as-is.
- **Unfreeze In View** (X/A button): Restores frozen voxels in your current frustum to normal integration.

This lets you selectively protect good surfaces while continuing to refine other areas.

### Training Gaussian Splats

Once the room is well-scanned:

1. Open the debug menu (left thumbstick click)
2. Verify the **Server URL** points to your PC running [RoomScan-GaussianSplatServer](https://github.com/arghyasur1991/RoomScan-GaussianSplatServer). If you used the setup wizard and your PC is on the same LAN, the IP is auto-detected and should already be correct. For a cloud/remote server, edit the URL in the debug menu or set it in the Inspector before building.
3. Press **Start GS Training** — this triggers the full pipeline automatically:
   - Exports the current mesh as a point cloud (`points3d.ply`)
   - ZIPs all keyframes, poses, and point cloud from `GSExport/`
   - Uploads the ZIP to the server
   - The debug menu shows live training status: state, progress bar, iteration count, elapsed time, backend
   - When training completes, the trained PLY is downloaded back to the Quest
4. Press **Render Mode** to cycle to Splat view — the downloaded PLY is loaded into `GaussianSplatRenderer` and rendered on-device
5. Cycle through Mesh → Splat → Both → Mesh to compare views

Scanning continues during training — you can keep refining the mesh while waiting.

### Saving and Loading

- **Save Scan**: Persists the full TSDF + color volumes, triplanar color textures, triplanar depth maps, and MRUK anchor pose to disk (`RoomScans/scan.bin` + `RoomScans/triplanar/`)
- **Load Scan**: Restores a previously saved scan. If the room anchor has moved since save, bakes the relocation into both the TSDF volume (compute resample) and triplanar textures (forward-splat compute) so the scan aligns with the physical room. Rebuilds the mesh. Validates that volume dimensions match.
- **Auto-save on quit**: If enabled, the scan is automatically saved when the app exits (default off)
- **Clear All Data**: Stops scanning, clears volumes/mesh/triplanar/keyframes, deletes saved scan and GSExport from disk

### Architecture

```
RoomAnchorManager (MRUK floor anchor → relocation matrix on load)
       │
PassthroughCameraProvider (RGB frames from headset cameras)
       │
DepthCapture (AROcclusionManager → depth → normals → dilation, tracking→world)
       │
VolumeIntegrator (TSDF + color integration, exclusion zones, prune, freeze, bake relocation)
       │
MeshExtractor → GPUSurfaceNets (compute: classify → smooth → snap → temporal → index)
       │         └── GPUMeshRenderer (Graphics.RenderPrimitivesIndirect, single draw call)
       │
       ├── TriplanarCache (bake camera RGB → 3 world-space textures + depth maps,
       │                    forward-splat relocation on load)
       │
       ├── KeyframeCollector (motion-gated JPEG + poses → GSExport/ on disk)
       │
       ├── PointCloudExporter (GPU mesh → points3d.ply via AsyncGPUReadback)
       │
       └── GSplatServerClient (ZIP upload → poll status → PLY download)
                   │
                   GSplatManager + GaussianSplatRenderer (UGS)
                   │
                   On-device Gaussian Splat rendering
```

See [ALGORITHM.md](ALGORITHM.md) for the full technical reference.

## Gaussian Splat Pipeline

QuestRoomScan captures keyframes and a dense point cloud during scanning, uploads them to a PC training server, and renders the trained Gaussian splats on-device. See [Usage Flow > Training Gaussian Splats](#training-gaussian-splats) for the step-by-step user guide.

### On-Device Capture (automatic during scanning)

- **KeyframeCollector**: Motion-gated JPEG frames + camera poses saved to `GSExport/` on disk (`images/*.jpg`, `frames.jsonl`). Captures are triggered by camera movement — you get more keyframes by looking at the room from different angles.
- **PointCloudExporter**: GPU mesh vertices exported as binary PLY (`points3d.ply`) via `AsyncGPUReadback`. Auto-exports every 30 seconds during scanning, or manually via the debug menu.

### Server Training (via [RoomScan-GaussianSplatServer](https://github.com/arghyasur1991/RoomScan-GaussianSplatServer))

The companion PC server handles the full training pipeline:

```bash
python main.py --port 8420  # API server
npm run dev                  # Dashboard at http://localhost:5173
```

When you press **Start GS Training** in the debug menu, the following happens automatically:

1. **Export**: Final point cloud exported from GPU mesh
2. **ZIP & Upload**: Quest packages `GSExport/` contents (`frames.jsonl`, `points3d.ply`, `images/*.jpg`) into a ZIP and POSTs to `{serverUrl}/upload?iterations={N}`
3. **Convert**: Server converts Unity poses + intrinsics to COLMAP binary format, computes scene normalization
4. **Train**: Gaussian Splat training via msplat (Metal), gsplat (CUDA), or 3DGS — the debug menu shows live progress
5. **Denormalize**: Output PLY is transformed back to world coordinates (reverses nerfstudio-style scene normalization)
6. **Download**: Quest GETs `{serverUrl}/download` → trained PLY bytes stored in memory
7. **View**: Press **Render Mode** to cycle to Splat — `GSplatManager` loads PLY via `GaussianSplatPlyLoader.LoadFromPlyBytes()` and renders on-device

### On-Device Rendering (UGS)

Trained splats are rendered using a [fork of Unity Gaussian Splatting](https://github.com/arghyasur1991/UnityGaussianSplatting) with runtime PLY loading and Quest 3 optimizations:

- **`GaussianSplatPlyLoader`**: Parses binary PLY → converts to UGS internal format → creates GPU buffers directly (no Editor asset pipeline needed)
- **Coordinate conversion**: COLMAP (right-handed Y-down) → Unity (left-handed Y-up)
- **Quest 3 stereo**: Per-eye VP matrices for correct VR covariance projection, shared compute between eyes
- **Performance**: Reduced-resolution rendering (0.5x), optimized compute shaders, partial radix sort, contribution-based culling
- **Render mode switching**: Mesh, Splat, or Both — cycled via debug menu or controller binding without releasing GPU resources

### Supported Training Backends

| Backend | Platform | Install |
|---------|----------|---------|
| [msplat](https://github.com/nicknish/msplat) | Apple Silicon (Metal) | `pip install "msplat[cli]"` |
| [gsplat](https://github.com/nerfstudio-project/gsplat) | NVIDIA GPU (CUDA) | `pip install gsplat` |
| [3DGS](https://github.com/graphdeco-inria/gaussian-splatting) | NVIDIA GPU (CUDA) | Clone repo, pass `--gs-repo` |

## VR Debug Menu

World-space UI Toolkit panel activated via **left thumbstick click** (Quest OS reserves the Menu/Start button for system use). Point the controller ray at buttons and press the **index trigger** to click.

The panel lazy-follows your gaze — it floats at 0.75m and re-centers when your head drifts past 45 degrees.

![Debug Menu](docs/debug-menu.png)

### Sections

**Scan Status** — Live readouts updated every frame:
- **Scanning**: Whether integration is active (Running / Stopped)
- **Mode**: Current scanning mode
- **Integrations**: Total depth frames integrated into the volume
- **Keyframes**: Number of motion-gated JPEG snapshots captured for GS training
- **Render**: Current render mode (Mesh / Splat / Both)

**Server Training** — Gaussian Splat training status:
- **Server**: Editable URL field pointing to your PC server (e.g. `http://192.168.1.100:8420`)
- **State**: Idle, Uploading, Training, Downloading, Done, Error
- **Progress**: Visual progress bar
- **Iteration**: Current / total training iterations
- **Elapsed**: Training wall-clock time
- **Backend**: Which training backend the server is using (msplat / gsplat / 3DGS)
- **Message**: Status messages from the server

**Persistence** — Data on disk:
- **Saved Scan**: Whether a saved scan exists (with size)
- **GSExport**: Whether keyframes/point cloud are on disk (with count)

### Action Buttons

| Button | What it does |
|--------|-------------|
| **Stop/Start Scanning** | Toggles depth integration, keyframe capture, and triplanar baking. Label updates to reflect current state. |
| **Render: Mesh** | Cycles render mode: Mesh → Splat → Both → Mesh. If a trained PLY has been downloaded but not yet loaded, switching to Splat auto-loads it. Label shows current mode. |
| **Save Scan** | Persists TSDF + color volumes, triplanar textures + depth maps, and room-anchor pose to disk. Button shows "Saving..." then "Done!" or "Failed". |
| **Load Scan** | Restores a previously saved scan, bakes room-anchor relocation if needed, and rebuilds the mesh. Validates volume dimensions match. |
| **Export Point Cloud** | Manually exports GPU mesh vertices as `points3d.ply` via async readback (also auto-exports every 30s during scanning). |
| **Start GS Training** | Triggers the full training pipeline: export point cloud → ZIP keyframes → upload → train → download. Disabled while training is in progress. |
| **Cancel Training** | Sends cancel request to the server. Only enabled while training is in progress. |
| **Clear All Data** | Stops scanning, clears all volumes/mesh/textures, deletes saved scan and GSExport from disk. |

**Footer**: Live FPS counter.

### Default Controller Bindings

| Button | Action |
|--------|--------|
| Left Thumbstick Click | Toggle Debug Menu |
| One (Y/B) | Freeze In View |
| Two (X/A) | Unfreeze In View |
| Three (A/X) | Cycle Render Mode |
| Four (B/Y) | Start Server Training (disabled by default) |

All bindings are configurable via `RoomScanInputHandler` — add, remove, or remap any `ScanAction` to any `OVRInput.Button`.

## Memory Budget (Quest 3)

Default values — all configurable per-component in the Inspector.

| Component | Default | Memory |
|-----------|---------|--------|
| TSDF volume (RG8_SNorm) | 256 x 256 x 256 | ~32 MB |
| Color volume (RGBA8) | 256 x 256 x 256 | ~64 MB |
| GPU Surface Nets (coord map, vertices, indices, smoothing, temporal 3D texture) | 256³ derived | ~83 MB |
| Triplanar color textures (3x RGBA8) | 3 x 4096 x 4096 | ~192 MB |
| Triplanar depth textures (3x R8) | 3 x 4096 x 4096 | ~48 MB |
| **Total GPU** | | **~419 MB** |

Keyframes are written as JPEGs to disk (not held in GPU memory). To reduce GPU memory on constrained devices, lower `VolumeIntegrator.voxelCount` and `TriplanarCache.textureResolution` in the Inspector.

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
| **Texturing** | Geometry only — no camera RGB texturing | Camera-based texturing: triplanar world-space cache (~8mm/texel) and vertex colors (~5cm) — sourced from passthrough camera RGB |
| **Persistence** | None — mesh lost on restart | Save/load of TSDF + color volumes + triplanar textures + depth maps to disk, MRUK room-anchor relocation on load |
| **Mesh quality** | Basic TSDF blending | Quality² modulation, confidence-gated Surface Nets, warmup clearing, pruning, body exclusion zones, GPU temporal stabilization, RANSAC plane detection & snapping |
| **Gaussian Splatting** | — | Full pipeline: on-device capture → PC server training → on-device UGS rendering with render mode switching |
| **VR UI** | — | World-space debug menu with controller ray interaction, live status, and training controls |
| **Packaging** | Embedded in a game | Standalone Unity package with one-click editor setup wizard |

## License

[MIT](LICENSE.md) — see [LICENSE.md](LICENSE.md) for full text and attribution.
