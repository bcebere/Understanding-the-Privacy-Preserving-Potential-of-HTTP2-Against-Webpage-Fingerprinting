# wfaudit relative
# h2deflib relative
from .core_defense import DEFENSE

CLMODS_DEFENSE = DEFENSE(
    name="all_mods",
    initial_window_size_strategy="random",  # "clwdn",
    request_batch=False,
    request_shuffle=True,
    random_pings=True,
    recv_interval_strategy=0.0001,
    recv_delay_threshold=10000,
    send_dummy_packet_strategy="random_batch",
    send_dummy_packet_limit=1,
    send_dummy_packet_loop=False,
)
