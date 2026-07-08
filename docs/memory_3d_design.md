# 3D Memory Graph Design Brief

## Goal

Build an experimental `/memory/3d` view for the Memory graph. The current `/memory` 2D WebGL view remains the compatibility and performance baseline. The 3D view should focus on exploration, spatial understanding, and the "cosmic / soap-bubble" feel the user wants.

## Current Entry Point

- Route: `/memory/3d`
- Template: `src/templates/memory_3d.html`
- Existing data APIs to reuse:
  - `GET /memory/graph/chunk?offset=...&limit=...`
  - `GET /memory/graph/meta`
  - `GET/PUT /memory/graph/snapshot`
  - `POST /memory/graph/status`
- Existing 2D implementation reference:
  - `src/templates/memory.html`

## Product Direction

Use a Three.js 3D graph where nodes behave like mass-bearing bubbles in space:

- Large / important nodes have larger radii and larger exclusion volumes.
- Small nodes can sit closer to each other and orbit around larger hubs.
- Edges are springs; force grows with stretch, so distant linked nodes pull harder.
- Collision/overlap avoidance is radius-based, not just visual-size-based.
- Hub nodes should not collapse their neighbors into one dense clump.

3D is acceptable and desired. Do not reject it just because projection overlap exists. Users can rotate the camera to resolve occlusion.

## Interaction Model

Use Blender-like navigation where practical:

- Middle mouse drag: orbit/rotate camera.
- Shift + middle mouse drag: pan.
- Mouse wheel: zoom.
- Left click: select node.
- Left drag selected node: move it on the current camera-facing plane or a ray-projected drag plane.
- Double click or focus button: fly camera to node.
- Selected node:
  - Highlight node.
  - Highlight adjacent edges.
  - Highlight neighbor nodes.
  - Fade non-neighborhood content.

Use `OrbitControls` from Three.js if available. Override mappings if needed to approximate Blender controls.

## Visual Model

- Render nodes as instanced spheres or billboarding impostors for performance.
- Depth should be readable:
  - Far nodes smaller and dimmer.
  - Edges fade with distance and non-selection.
  - Optional fog, but avoid making the graph too blurry.
- Use clear node boundaries. Nodes must feel like individual objects, not a cloud.
- Labels should be sparse:
  - Show selected / hovered / search hit labels.
  - Optionally show nearby labels only when zoomed in.
  - Avoid rendering thousands of DOM labels every frame.

## Physics Model

The 3D model should not be naive O(n^2) full repulsion per frame for large graphs.

Recommended approach:

- Store each node position and velocity as `{ x, y, z, vx, vy, vz }`.
- Compute mass/radius from:
  - visual size,
  - degree,
  - node type,
  - future importance score if available.
- Radius formula can start simple:
  - `radius = baseRadius * (1 + log1p(degree) * 0.18) * groupFactor`
- Forces:
  - Center gravity, weak and configurable.
  - Edge springs:
    - `stretch = distance - restLength`
    - `force = stretch * springConstant`
    - restLength increases for high-degree hubs and large-radius pairs.
  - Bubble collision:
    - if `distance < radiusA + radiusB + padding`, push apart strongly.
  - Long-range repulsion:
    - sampled, spatial-grid-based, or Barnes-Hut-like approximation.
    - fixed per-frame force budget for 1000+ nodes.
- Damping:
  - Use velocity damping to prevent endless jitter.
  - Avoid freezing immediately after load; large graphs need longer settling.

## Performance Requirements

Target graph size based on current local data:

- About 4,236 nodes.
- About 6,398 edges.

Performance target:

- Interactive camera movement should stay smooth.
- Physics should not block camera rendering.
- Cap physics work per frame.
- Consider separating render FPS and physics tick budget.

Implementation preferences:

- Use Three.js.
- Prefer `InstancedMesh` for nodes.
- Use `BufferGeometry` for edges.
- Avoid rebuilding all geometry every frame when only positions changed; update buffers.
- Use requestAnimationFrame loop with adaptive physics steps.

## Snapshot Requirements

3D snapshots must not reuse incompatible 2D positions.

Save:

- `layoutMode: "3d"`
- `layoutVersion`
- node positions: `{ x, y, z }`
- optional velocities if useful
- camera state:
  - position
  - target
  - zoom / distance

Keep 2D and 3D snapshot keys or layout versions separate.

## UI Scope For First Prototype

The first prototype does not need every 2D feature. It should include:

- Load graph chunks from existing API.
- Render 3D nodes and edges.
- Orbit/pan/zoom controls.
- Click select node.
- Highlight selected node neighborhood.
- Basic status HUD: node count, edge count, physics state.
- Freeze / resume physics button.
- Fit / focus camera button.

Do not spend the first prototype on:

- Full search parity.
- Full side-detail parity.
- Complex label rendering.
- Snapshot migration from 2D.

## Existing Page State

`src/templates/memory.html` has an in-progress custom WebGL 2D implementation. It has become complex and should be treated as a reference, not copied wholesale. Reuse data normalization ideas where helpful, but build the 3D view independently.

## Suggested Milestones

1. Build static Three.js scene on `/memory/3d` with test nodes.
2. Load real graph chunks and place nodes in deterministic initial 3D positions.
3. Render nodes as instanced spheres and edges as line buffer geometry.
4. Add orbit controls and picking.
5. Add 3D bubble/spring physics with fixed per-frame budget.
6. Add highlighting, labels for selected/hovered nodes, and camera focus.
7. Add 3D snapshot save/load with separate layout version.
