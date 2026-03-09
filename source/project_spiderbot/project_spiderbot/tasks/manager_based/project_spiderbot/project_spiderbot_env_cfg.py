# SPDX-License-Identifier: BSD-3-Clause
# Clean Spiderbot manager-based RL env cfg (event-driven streak curriculum + curriculum apply)

from __future__ import annotations

import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets import ArticulationCfg, AssetBaseCfg
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.envs.mdp import observations as mdp_obs
from isaaclab.envs.mdp import rewards as mdp_rew
from isaaclab.envs.mdp import terminations as mdp_term
from isaaclab.envs.mdp import events as mdp_events
from isaaclab.envs.mdp import modify_term_cfg
from isaaclab.envs.mdp.actions import JointPositionActionCfg
from isaaclab.envs.mdp.commands import UniformVelocityCommandCfg
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.managers import CurriculumTermCfg as CurrTerm
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.utils import configclass

from . import mdp as spiderbot_mdp
from .mdp import spiderbot_curriculums as spiderbot_curr


# -----------------------------------------------------------------------------
# Source of truth USD for spiderbot RL
# -----------------------------------------------------------------------------
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


# -----------------------------------------------------------------------------
# Scene
# -----------------------------------------------------------------------------
@configclass
class SpiderbotSceneCfg(InteractiveSceneCfg):
    """Minimal scene: ground + light + spiderbot articulation."""

    ground = AssetBaseCfg(
        prim_path="/World/ground",
        spawn=sim_utils.GroundPlaneCfg(size=(100.0, 100.0)),
    )

    dome_light = AssetBaseCfg(
        prim_path="/World/DomeLight",
        spawn=sim_utils.DomeLightCfg(intensity=500.0),
    )

    robot: ArticulationCfg = ArticulationCfg(
        prim_path="{ENV_REGEX_NS}/Robot",
        spawn=sim_utils.UsdFileCfg(
            usd_path=SPIDERBOT_USD_PATH,
            activate_contact_sensors=False,
        ),
        actuators={
            "legs": ImplicitActuatorCfg(
                joint_names_expr=SPIDERBOT_LEG_JOINTS,
                stiffness=40.0,
                damping=2.0,
            )
        },
    )


# -----------------------------------------------------------------------------
# Actions / Commands
# -----------------------------------------------------------------------------
@configclass
class ActionsCfg:
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
        resampling_time_range=(1.0, 3.0),
        ranges=UniformVelocityCommandCfg.Ranges(
            lin_vel_x=(0.0, 0.0),  # stage 0 default; curriculum apply overrides after stage changes
            lin_vel_y=(0.0, 0.0),
            ang_vel_z=(0.0, 0.0),
        ),
    )


# -----------------------------------------------------------------------------
# Observations
# -----------------------------------------------------------------------------
@configclass
class ObservationsCfg:
    """Minimal observations for velocity-tracking RL."""

    @configclass
    class PolicyCfg(ObsGroup):
        leg_joint_pos = ObsTerm(
            func=mdp_obs.joint_pos_rel,
            params={"asset_cfg": SceneEntityCfg("robot", joint_names=SPIDERBOT_LEG_JOINTS)},
        )
        leg_joint_vel = ObsTerm(
            func=mdp_obs.joint_vel_rel,
            params={"asset_cfg": SceneEntityCfg("robot", joint_names=SPIDERBOT_LEG_JOINTS)},
        )

        base_lin_vel = ObsTerm(func=mdp_obs.base_lin_vel)
        base_ang_vel = ObsTerm(func=mdp_obs.base_ang_vel)

        commanded_vel = ObsTerm(func=mdp_obs.generated_commands, params={"command_name": "base_velocity"})

        def __post_init__(self) -> None:
            self.enable_corruption = False
            self.concatenate_terms = True

    policy: PolicyCfg = PolicyCfg()


# -----------------------------------------------------------------------------
# Events (Reset + interval curriculum updater)
# -----------------------------------------------------------------------------
@configclass
class EventsCfg:
    reset_scene_to_default = EventTerm(
        func=mdp_events.reset_scene_to_default,
        mode="reset",
    )

    # Every-step interval: updates stage/streak and logs metrics inside spiderbot_curr.update_streak_every_step
    # IMPORTANT: do NOT pass env_ids in params; EventManager supplies env_ids automatically.
    curriculum_update = EventTerm(
        func=spiderbot_curr.update_streak_every_step,
        mode="interval",
        interval_range_s=(0.0, 0.0),
        params={
            "tilt_limit_angle": 1.80,
            "tag": "spiderbot_curriculum",
        },
    )


# -----------------------------------------------------------------------------
# Curriculum (only APPLY stage -> lin_vel_x range)
# -----------------------------------------------------------------------------
@configclass
class CurriculumCfg:
    lin_vel_x_apply = CurrTerm(
        func=modify_term_cfg,
        params={
            "address": "commands.base_velocity.ranges.lin_vel_x",
            "modify_fn": spiderbot_curr.lin_vel_x_from_stage,
            "modify_params": {},  # uses spiderbot_curr.STAGES internally
        },
    )


# -----------------------------------------------------------------------------
# Rewards / Terminations
# -----------------------------------------------------------------------------
@configclass
class RewardsCfg:
    """Stand-stage rewards (cmd_vel == 0 initially)."""

    track_lin_vel_xy = RewTerm(
        func=mdp_rew.track_lin_vel_xy_exp,
        weight=1.0,
        params={"command_name": "base_velocity", "std": 0.25},
    )

    track_ang_vel_z = RewTerm(
        func=mdp_rew.track_ang_vel_z_exp,
        weight=0.0,
        params={"command_name": "base_velocity", "std": 0.25},
    )

    action_rate_l2 = RewTerm(func=mdp_rew.action_rate_l2, weight=-0.001)

    joint_vel_l2 = RewTerm(
        func=mdp_rew.joint_vel_l2,
        weight=-0.0005,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=SPIDERBOT_LEG_JOINTS)},
    )

    upright = RewTerm(func=spiderbot_mdp.upright_posture, weight=0.5)

    yaw_rate_l2 = RewTerm(func=spiderbot_mdp.ang_vel_z_l2, weight=-0.2)

    stand_joint_pos_l2 = RewTerm(
        func=spiderbot_mdp.joint_pos_target_l2,
        weight=-0.5,
        params={
            "target": 0.0,
            "asset_cfg": SceneEntityCfg(name="robot", joint_names=SPIDERBOT_LEG_JOINTS),
        },
    )


@configclass
class TerminationsCfg:
    time_out = DoneTerm(func=mdp_term.time_out, time_out=True)
    bad_orientation = DoneTerm(func=spiderbot_mdp.bad_orientation_quat, params={"limit_angle": 1.80})


# -----------------------------------------------------------------------------
# Main EnvCfg
# -----------------------------------------------------------------------------
@configclass
class ProjectSpiderbotEnvCfg(ManagerBasedRLEnvCfg):
    scene: SpiderbotSceneCfg = SpiderbotSceneCfg()
    actions: ActionsCfg = ActionsCfg()
    commands: CommandsCfg = CommandsCfg()
    observations: ObservationsCfg = ObservationsCfg()
    events: EventsCfg = EventsCfg()
    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()
    curriculum: CurriculumCfg = CurriculumCfg()

    def __post_init__(self) -> None:
        self.sim.dt = 1.0 / 120.0
        self.decimation = 2
        self.episode_length_s = 10.0

        self.scene.num_envs = 32
        self.scene.env_spacing = 4.0