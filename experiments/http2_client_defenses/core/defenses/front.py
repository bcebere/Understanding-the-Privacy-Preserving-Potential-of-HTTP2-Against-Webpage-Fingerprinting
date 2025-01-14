# wfaudit relative
from .core_defense import DEFENSE

FRONT_DEFENSE = DEFENSE(
    name="front",
    send_dummy_packet_strategy="random_per_frame",
)
