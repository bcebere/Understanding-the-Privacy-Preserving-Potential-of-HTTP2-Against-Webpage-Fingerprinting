# wfaudit relative
from .core_defense import DEFENSE

TAMARAW_DEFENSE = DEFENSE(
    name="tamaraw",
    initial_window_size_strategy="random",  # "constant",
    send_packet_size_strategy="random_per_connection",
    send_interval_strategy="random_per_connection",
    send_dummy_packet_strategy="random_per_connection",
)
