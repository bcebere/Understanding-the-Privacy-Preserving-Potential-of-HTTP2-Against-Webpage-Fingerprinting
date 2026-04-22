# h2deflib relative
from .core_defense import DEFENSE

FRONT_DEFENSE = DEFENSE(
    name="front",
    send_dummy_packet_strategy="front",
    send_dummy_packet_interval="front",
    send_dummy_packet_loop=True,
)
