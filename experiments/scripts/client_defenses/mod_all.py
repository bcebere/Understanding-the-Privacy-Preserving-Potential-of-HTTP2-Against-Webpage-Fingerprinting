# wfaudit relative
from .core_defense import DEFENSE

CLMODS_DEFENSE = DEFENSE(
    name="all_mods",
    initial_window_size_strategy="random",  # "clwdn",
    request_batch=True,  # batching
    request_shuffle=True,  # batching
    random_pings=True,
    recv_interval_strategy="random_per_frame",
    send_dummy_packet_strategy="random_batch",
    send_dummy_packet_limit=3,
    send_dummy_packet_loop=False,
)
