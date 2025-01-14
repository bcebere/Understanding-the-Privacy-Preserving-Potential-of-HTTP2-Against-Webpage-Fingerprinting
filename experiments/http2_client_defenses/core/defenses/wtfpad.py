# wfaudit relative
from .core_defense import DEFENSE

WTFPAD_DEFENSE = DEFENSE(
    name="wtfpad",
    initial_window_size_strategy="random",  # "constant",
    send_packet_size_strategy="random_per_frame",
    send_interval_strategy="random_per_frame",
    send_dummy_packet_strategy="random_per_frame",
)
