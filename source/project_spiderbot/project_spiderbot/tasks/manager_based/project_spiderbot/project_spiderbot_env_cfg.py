# SPDX-License-Identifier: BSD-3-Clause
# Clean Spiderbot manager-based RL env cfg (no cartpole remnants)

import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets import ArticulationCfg, AssetBaseCfg
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.utils import configclass

from . import mdp

# Source of truth USD for spiderbot RL
SPIDERBOT_USD_PATH = (
    "/home/harsh/work/Applied_Project-Spider_Bot/src/prjct_spider_bot_description/Isaac_GUI/spiderbot_rl_nophysics.usd"
)


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
    # NOTE: IsaacLab requires actuators to be specified. For now we attach a minimal implicit actuator
    # to *all* joints. Next phase we'll restrict to the 12 leg joints and exclude any sensor joints.
    robot: ArticulationCfg = ArticulationCfg(
        prim_path="{ENV_REGEX_NS}/Robot",
        spawn=sim_utils.UsdFileCfg(
            usd_path=SPIDERBOT_USD_PATH,
            activate_contact_sensors=False,
        ),
        actuators={
            "all_joints": ImplicitActuatorCfg(
                joint_names_expr=[".*"],
                stiffness=0.0,
                damping=0.0,
            )
        },
    )


@configclass
class ActionsCfg:
    """Placeholder. We'll add 12-leg joint actions next."""
    pass


@configclass
class ObservationsCfg:
    """Placeholder. We'll add joint/base/command observations next."""

    @configclass
    class PolicyCfg(ObsGroup):
        def __post_init__(self) -> None:
            self.enable_corruption = False
            self.concatenate_terms = False

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