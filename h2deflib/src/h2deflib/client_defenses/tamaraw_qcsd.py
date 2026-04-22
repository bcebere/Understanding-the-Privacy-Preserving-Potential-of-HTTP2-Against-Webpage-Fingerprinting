# h2deflib relative
from .core_defense import DEFENSE

TAMARAW_QCSD_DEFENSE = DEFENSE(
    name="tamaraw",
    initial_window_size_strategy=4096,  # "constant",
    recv_delay_threshold=4096,
    recv_interval_strategy=0.01,
    send_dummy_packet_strategy="random_per_connection",
    send_dummy_packet_interval=0.02,
    send_dummy_packet_loop=True,
)
