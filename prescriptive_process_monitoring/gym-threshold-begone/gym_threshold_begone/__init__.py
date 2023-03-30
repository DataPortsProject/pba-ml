from gym.envs.registration import register

register(
    id='bpm2020-v0',
    entry_point='gym_threshold_begone.envs:BPM2020Env',
)
register(
    id='bpm2020fixed-v0',
    entry_point='gym_threshold_begone.envs:BPM2020EnvFixedProcesslength',
)
register(
    id='non_prophetic_curiosity-v0',
    entry_point='gym_threshold_begone.envs:NonPropheticCuriosity',
)
