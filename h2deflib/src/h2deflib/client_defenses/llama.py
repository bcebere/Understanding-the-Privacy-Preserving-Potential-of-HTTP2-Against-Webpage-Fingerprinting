# wfaudit relative
# h2deflib relative
from .core_defense import DEFENSE

LLAMA_DEFENSE = DEFENSE(
    name="llama",
    send_dummy_packet_strategy="llama",
    send_dummy_packet_interval="llama",
    request_delay=True,
    request_batch=True,
    request_shuffle=True,
)
