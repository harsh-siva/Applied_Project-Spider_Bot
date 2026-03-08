# SPDX-License-Identifier: BSD-3-Clause
# Clean Spiderbot manager-based RL env cfg (no cartpole remnants)

import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets import ArticulationCfg, AssetBaseCfg
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.utils import configclass
from isaaclab.envs.mdp.actions import JointPositionActionCfg
from isaaclab.envs.mdp.commands import UniformVelocityCommandCfg

from . import mdp

# Source of truth USD for spiderbot RL
SPIDERBOT_USD_PATH = (
    "/home/harsh/work/Applied_Project-Spider_Bot/src/prjct_spider_bot_description/Isaac_GUI/spiderbot_rl_nophysics.usd"
)

# Lock 12 leg joints (exclude J_Lidar)
SPIDERBOT_LEG_JOINTS = [
    "J_Coxa_BL",
    "J_Coxa_BR",
    "J_Coxa_FL",
    "J_Coxa_FR",
    "J_Femur_BL",
    "J_Femur_BR",
    "J_Femur_FL",
    "J_Femur_FR",
    "J_Tibia_BL",
    "J_Tibia_BR",
    "J_Tibia_FL",
    "J_Tibia_FR",
]


@configclass
class SpiderbotSceneCfg(InteractiveSceneCfg):
    """Minimal scene: ground + light + spiderbot articulation."""

    # ground plane
    ground = AssetBaseCfg(
        prim_path="/World/ground",
        spawn=sim_utils.GroundPlaneCfg(size=(100.0, 100.0)),
    )

    # lights
    dome_light = AssetBaseCfg(
        prim_path="/World/DomeLight",
        spawn=sim_utils.DomeLightCfg(intensity=500.0),
    )

    # Spawn the spiderbot USD as an articulation
    robot: ArticulationCfg = ArticulationCfg(
        prim_path="{ENV_REGEX_NS}/Robot",
        spawn=sim_utils.UsdFileCfg(
            usd_path=SPIDERBOT_USD_PATH,
            activate_contact_sensors=False,
        ),
        actuators={
            "legs": ImplicitActuatorCfg(
                joint_names_expr=SPIDERBOT_LEG_JOINTS,
                stiffness=0.0,
                damping=0.0,
            )
        },
    )


@configclass
class ActionsCfg:
    """Action terms for the environment."""
    legs_joint_pos = JointPositionActionCfg(
        asset_name="robot",
        joint_names=SPIDERBOT_LEG_JOINTS,
        scale=0.25,
        use_default_offset=True,
    )


@configclass
class CommandsCfg:
    base_velocity = UniformVelocityCommandCfg(
        asset_name="robot",
        resampling_time_range=(1000.0, 1000.0),  # effectively fixed during smoke
        ranges=UniformVelocityCommandCfg.Ranges(
            lin_vel_x=(-0.5, 0.5),
            lin_vel_y=(-0.3, 0.3),
            ang_vel_z=(-1.0, 1.0),
        ),
    )


@configclass
class ObservationsCfg:
    """Minimal observations. We'll add joint/base state next."""

    @configclass
    class PolicyCfg(ObsGroup):
        # IMPORTANT: observation terms must be ObservationTermCfg, not raw functions
        commanded_vel = ObsTerm(func=mdp.generated_commands, params={"command_name": "base_velocity"})

        def __post_init__(self) -> None:
            self.enable_corruption = False
            self.concatenate_terms = True

    policy: PolicyCfg = PolicyCfg()


@configclass
class RewardsCfg:
    """Placeholder."""
    pass


@configclass
class TerminationsCfg:
    """Minimal termination set required by ManagerBasedRLEnvCfg validation."""
    time_out = DoneTerm(func=mdp.time_out, time_out=True)


@configclass
class ProjectSpiderbotEnvCfg(ManagerBasedRLEnvCfg):
    """Top-level env cfg."""
    scene: SpiderbotSceneCfg = SpiderbotSceneCfg()
    actions: ActionsCfg = ActionsCfg()
    commands: CommandsCfg = CommandsCfg()
    observations: ObservationsCfg = ObservationsCfg()
    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()

    def __post_init__(self) -> None:
        # Basic sim step sizes (safe defaults for smoke tests; tune later)
        self.sim.dt = 1.0 / 120.0
        self.decimation = 2

        # Required fields
        self.episode_length_s = 10.0

        # Scene layout
        self.scene.num_envs = 1
        self.scene.env_spacing = 4.0