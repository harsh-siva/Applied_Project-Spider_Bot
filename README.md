# Applied Project — Spider Bot (Checkpoint 1)

## What works at this checkpoint
- ROS 2 workspace: `prjct_spiderbot_ws`
- Package: `prjct_spider_bot_description`
- RViz visualization via `display.launch.py`

## Reproduce RViz visualization (ROS 2 Humble)
```bash
cd ~/prjct_spiderbot_ws
source /opt/ros/humble/setup.bash
colcon build --symlink-install
source install/setup.bash
ros2 launch prjct_spider_bot_description display.launch.py
```

## Notes
- URDF/Xacro is treated as the single source of truth for the robot model.
- Sensor frames exist in the URDF; Isaac Sim sensors will attach to those frames in USD later.
## Checkpoint 4 — Proper collision meshes (VHACD per link) + locked sensor frames

This checkpoint replaces primitive collision geometry with **per-link convex decomposition** meshes generated using **VHACD** (via PyBullet), and locks common sensor frames (lidar/camera) so TF is stable.

### Why not use visual meshes as collisions?
Visual meshes are often:
- high triangle count (slow for collision checking / physics),
- non-convex (many engines handle non-convex poorly),
- unstable in simulation (can cause jitter).

Instead, we use **convex hull approximations** per link.

### What we did

#### 1) Generated VHACD collision meshes
- Converted each `*.stl` to `*.obj` (temporary) because the VHACD tool expects OBJ.
- Ran VHACD to generate convex decomposition collision meshes:
  - Output: `meshes/collision/*_vhacd.obj`
  - Logs: `meshes/collision/*_vhacd_log.txt`

> Note: `.venv_collision/` and `meshes/obj_tmp/` are tooling artifacts and should NOT be committed to git.

#### 2) Updated the URDF/xacro to use VHACD meshes for `<collision>`
All `<collision>` blocks now point to:
`meshes/collision/<linkname>_vhacd.obj`

Example:
```xml
<collision>
  <origin xyz="0 0 0" rpy="0 0 0"/>
  <geometry>
    <mesh filename="file://$(find prjct_spider_bot_description)/meshes/collision/base_link_vhacd.obj"
          scale="0.001 0.001 0.001"/>
  </geometry>
</collision>
