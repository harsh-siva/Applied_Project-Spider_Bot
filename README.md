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
