from .custom_car_racing import CustomCarRacing

from gymnasium import register

register(
    id='InternalStateCarRacing-v0',
    entry_point=f'{__name__}:CustomCarRacing',
    max_episode_steps=1000,
    disable_env_checker=True,
)
