import gymnasium as gym

from .custom_env_direct import UR5eRG2CustomTableEnv
from .custom_env_direct_cfg import UR5eRG2CustomTableEnvCfg

##
# Register Gym environments.
##

gym.register(
    id="Isaac-UR5e-RG2-Custom-Direct-v0",
    entry_point="isaaclab_tasks.direct.custom_task_direct:UR5eRG2CustomTableEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": UR5eRG2CustomTableEnvCfg,
    },
)

"""gym.register(
    id="Isaac-UR5e-RG2-Custom-Direct-v0",
    entry_point=f"{__name__}.custom_env_direct:UR5eRG2CustomTableEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.custom_env_direct:UR5eRG2CustomTableEnvCfg",
    },
)"""