# QuestRoomScan

Real-time 3D room reconstruction on Meta Quest 3. Produces a textured mesh from depth + RGB camera data using GPU TSDF volume integration and Surface Nets mesh extraction, with server-based Gaussian Splat training and on-device rendering via [Unity Gaussian Splatting](https://github.com/arghyasur1991/UnityGaussianSplatting).

## Features

- **GPU TSDF Integration** — Depth frames fused into a signed distance field via compute shaders
- **GPU Surface Nets Meshing** — Fully GPU-driven mesh extraction via compute shaders with zero CPU readback, rendered via a single `Graphics.RenderPrimitivesIndirect` draw call
- **Three-Layer Texturing** — Triplanar world-space cache (persistent ~8mm/texel) → vertex colors (~5cm fallback) — all sourced from passthrough camera RGB. Keyframes captured as motion-gated JPEGs to disk for Gaussian Splat training.
- **Package-Based Persistence** — Multi-scan persistence system where each scan is a self-contained package (`pkg_YYYYMMDD_HHMMSS/`) with its own TSDF, triplanar textures, keyframes, splat, and refined textures. Scan browser in the debug menu lists all saved packages. Artifacts (splat, refined, HQ) auto-save to the active package on creation.
- **OVRSpatialAnchor Relocation** — `RoomAnchorManager` creates a persisted `OVRSpatialAnchor` per scan package for reliable cross-session relocation. Per-artifact creation matrices in `anchor.json` track when each artifact was created relative to the spatial anchor, enabling accurate relocation even for artifacts created across different sessions. Falls back to MRUK floor anchor if spatial anchor localization fails.
- **Temporal Stabilization** — Adaptive per-vertex temporal blending on GPU prevents mesh jitter while allowing fast convergence
- **Exclusion Zones** — Cylindrical rejection around tracked heads prevents body reconstruction (configurable radius and height, up to 64 zones)
- **Gaussian Splat Training & Rendering** — Keyframe capture + point cloud export → PC server training → trained PLY download → on-device UGS rendering
- **VR Debug Menu** — Two-panel world-space UI Toolkit HUD with left navigation (Scan, Saved Scans, Refine, Training, Tools) and right detail views. Includes scan browser with load/delete per package, context-sensitive artifact deletion, and dynamic button disabled states.
- **Texture Refinement** — Post-scan texture refinement using captured keyframes. GPU compute shader bakes a UV atlas from the best-scoring keyframe projections per texel, with occlusion-aware depth testing and atomic score selection. Produces significantly sharper textures than the real-time triplanar cache. Optional server-side HQ refinement via differentiable rendering (currently experimental/broken).
- **Render Mode Switching** — Cycle between Mesh, Textured, Refined, HQRefined, and Splat views at runtime via debug menu or controller binding (default: A/X button)

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

### Texture Refinement

After scanning, you can produce a sharper UV-mapped texture atlas from the captured keyframes:

1. Open the debug menu
2. Press **Refine Textures** — this runs the full on-device pipeline:
   - **GPU readback**: Reads the current mesh from the GPU Surface Nets buffers
   - **Mesh simplification** (optional): meshoptimizer reduces triangle count (default 50%) to speed up unwrapping
   - **UV unwrapping**: xatlas (native C++ via P/Invoke) generates a UV atlas with seam-aware parameterization, with tunable chart/pack options for speed vs quality
   - **GPU atlas baking**: A compute shader (`AtlasBakeCompute.compute`) processes each keyframe — builds an occlusion depth buffer, then rasterizes UV-space triangles with per-texel keyframe projection, occlusion testing, and atomic best-score selection (~5-10s for 300 keyframes)
   - **Dilation**: Fills gaps at UV island edges
3. Press **Render Mode** to cycle to **Refined** — the UV-mapped mesh with baked atlas texture replaces the triplanar view

The refined texture is significantly sharper than the real-time triplanar cache because it selects the single best keyframe per texel rather than blending multiple noisy projections.

**HQ Refine (Server)** is also available in the debug menu — it uploads the UV-unwrapped mesh and keyframes to the server for differentiable texture optimization. **Note: server-side refinement is currently broken** (produces incorrect atlas output). On-device refinement is the recommended path.

Refined textures persist with **Save Scan** and are restored on **Load Scan**.

### Saving and Loading

QuestRoomScan uses a **package-based persistence system**. Each scan is saved as a self-contained package under `RoomScans/`:

```
RoomScans/
  manifest.json
  pkg_20260228_143022/
    scan.bin              # TSDF + color volumes (v1 binary)
    anchor.json           # Spatial anchor UUID + per-artifact matrices
    triplanar/            # Color + depth textures
    keyframes/            # Copied from GSExport/ (images + frames.jsonl)
    splat.ply             # Auto-saved when GS training completes
    refined_mesh.bin      # Auto-saved when on-device refinement completes
    refined_atlas.raw     # Auto-saved with refined mesh
    hq_atlas.raw          # Auto-saved when HQ refinement completes
```

- **Save Scan**: Creates a new package. Persists the TSDF + color volumes, triplanar textures, copies keyframes, and creates a persisted `OVRSpatialAnchor` for cross-session relocation. Sets this package as the active target for subsequent artifact auto-saves.
- **Load Scan**: Browse saved packages in the **Saved Scans** view. Loading a package localizes the spatial anchor, computes per-artifact relocation matrices, and restores all data including splat, refined textures, and HQ atlas.
- **Auto-save artifacts**: When a splat download completes, on-device refinement finishes, or HQ refinement finishes, the artifact is automatically saved to the active package — no manual "Save Scan" needed.
- **Delete artifact**: Context-sensitive deletion in the Scan view — deletes the artifact matching the current render mode (splat, refined, or HQ) from the active package.
- **Delete package**: Full package deletion from the Saved Scans view, including erasing the spatial anchor from persistent storage.
- **Clear All Data**: Stops scanning, clears volumes/mesh/triplanar/keyframes from memory, clears the active package reference.

### Architecture

```
RoomAnchorManager (OVRSpatialAnchor persistence + MRUK fallback → per-artifact relocation)
       │
RoomScanPersistence (package-based multi-scan persistence, anchor.json, manifest.json)
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
       ├── GSplatServerClient (ZIP upload → poll status → PLY download)
       │               │
       │               GSplatManager + GaussianSplatRenderer (UGS)
       │               │
       │               On-device Gaussian Splat rendering
       │
       └── TextureRefinement (GPU readback → xatlas UV unwrap → compute shader atlas bake)
                   │
                   RefinedMesh.shader (UV-mapped atlas rendering)
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

Two-panel world-space UI Toolkit panel activated via **left thumbstick click**. Point the controller ray at buttons and press the **index trigger** to click. The panel lazy-follows your gaze at 0.75m.

### Layout

```
+------------------+---------------------------------------------+
| QUESTROOMSCAN    |  [Right panel — swaps based on nav]          |
|  DEBUG           |                                             |
|                  |                                             |
| [*] Scan         |  (Scan View / Saved Scans / Refine /       |
| [ ] Saved Scans  |   Training / Tools)                         |
| [ ] Refine       |                                             |
| [ ] Training     |                                             |
| [ ] Tools        |                                             |
|                  |                                             |
| 72 FPS           |                                             |
+------------------+---------------------------------------------+
```

### Views

**Scan** (default) — Live status rows (Scanning, Mode, Integrations, Keyframes, Render, Package) and action buttons:
- **Start/Stop Scanning**: Toggle depth integration
- **Render Mode**: Cycle through Mesh → Textured → Refined → HQRefined → Splat
- **Save Scan**: Create a new package with current scan data
- **Delete Artifact**: Context-sensitive — deletes Splat/Refined/HQ atlas from active package based on current render mode

**Saved Scans** — Scrollable list of saved packages sorted newest-first. Each entry shows display name, date, artifact badges (KF, Splat, Refined, HQ), and Load/Delete buttons. Badge count shown on the nav button.

**Refine** — On-device and server refinement status + action buttons:
- **Refine Textures**: On-device GPU atlas bake from keyframes
- **HQ Refine (Server)**: Upload to server for differentiable optimization

**Training** — GS training with server URL field, live progress, and Start/Cancel buttons.

**Tools** — Export Point Cloud, Clear All Data.

### Button Disabled States

Buttons are dynamically enabled/disabled based on app context:
- **Save Scan**: Disabled if no volume data
- **Start GS Training**: Disabled if already training or no scan loaded
- **Refine Textures**: Disabled if already refining or no mesh/keyframes
- **HQ Refine**: Disabled if no server URL configured
- **Export Point Cloud**: Disabled if no volume data
- **Delete Artifact**: Only visible in Splat/Refined/HQRefined modes, requires active package

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

The texture refinement pipeline uses two open-source native C++ libraries:

- **[xatlas](https://github.com/jpcy/xatlas)** by Jonathan Young (MIT) — automatic UV atlas generation with seam-aware chart parameterization and efficient packing. Used for UV unwrapping the GPU Surface Nets mesh prior to atlas baking.
- **[meshoptimizer](https://github.com/zeux/meshoptimizer)** v1.0 by Arseny Kapoulkine (MIT) — mesh optimization toolkit. The `meshopt_simplify` function is used for optional mesh decimation before UV unwrapping, reducing triangle count while preserving topology to speed up the xatlas charting and packing phases.

Both libraries are compiled into a single native shared library (`libxatlas.so` / `libxatlas.bundle`) and invoked via P/Invoke from C#.

QuestRoomScan builds on that foundation with significant extensions:

| | lasertag | QuestRoomScan |
|-|----------|---------------|
| **Mesh extraction** | CPU marching cubes from GPU volume | Fully GPU-driven Surface Nets via compute shaders — zero CPU readback, single indirect draw call |
| **Texturing** | Geometry only — no camera RGB texturing | Camera-based texturing: triplanar world-space cache (~8mm/texel) and vertex colors (~5cm) — sourced from passthrough camera RGB |
| **Persistence** | None — mesh lost on restart | Multi-scan package persistence with OVRSpatialAnchor cross-session relocation |
| **Mesh quality** | Basic TSDF blending | Quality² modulation, confidence-gated Surface Nets, warmup clearing, pruning, body exclusion zones, GPU temporal stabilization, RANSAC plane detection & snapping |
| **Gaussian Splatting** | — | Full pipeline: on-device capture → PC server training → on-device UGS rendering with render mode switching |
| **VR UI** | — | World-space debug menu with controller ray interaction, live status, and training controls |
| **Packaging** | Embedded in a game | Standalone Unity package with one-click editor setup wizard |

## License

[MIT](LICENSE.md) — see [LICENSE.md](LICENSE.md) for full text and attribution.
