# Quest Room Scan

On-device continuous room scanning for Quest 3. Produces a vertex-colored mesh
from depth + RGB camera data using GPU TSDF volume integration and Surface Nets
mesh extraction.

## Requirements

- Unity 6 (6000.x)
- URP (Universal Render Pipeline)
- Quest 3 device

### Runtime dependencies (provided by your project)

- `com.unity.xr.meta-openxr` 2.1+ (bridges Meta's depth to AR Foundation)
- `com.unity.xr.openxr` 1.15+

### Android permissions

- `com.oculus.permission.USE_SCENE` (depth API)
- `android.permission.CAMERA` (RGB camera access)

## Installation

Add to your project's `Packages/manifest.json`:

```json
"com.genesis.roomscan": "file:/path/to/QuestRoomScan/package"
```

## Quick Start

1. Add `AROcclusionManager` to your XR Camera Rig
2. Add a `RoomScanner` component to a GameObject in your scene
3. Assign the compute shaders from this package
4. Build and deploy to Quest 3

## Architecture

Adapted from [anaglyphs/lasertag](https://github.com/anaglyphs/lasertag) (MIT).

- **DepthCapture**: Wraps AR Foundation `AROcclusionManager` for depth frames
- **VolumeIntegrator**: GPU TSDF volume integration via compute shaders
- **ChunkManager / SurfaceNetsMesher**: Chunk-based async mesh extraction (Burst Jobs)
- **TextureProjector**: Per-vertex color projection from RGB camera frames
- **RoomScanner**: Top-level orchestrator (passive + guided scan modes)
