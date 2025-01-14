# wfaudit relative
from .core_defense import DEFENSE

ADAPTIVE_DEFENSE = DEFENSE(
    name="adaptive",
    initial_window_size_strategy="random",  # "constant",
    send_packet_size_strategy="disabled",
    send_interval_strategy="adaptive",
    send_dummy_packet_strategy="adaptive",
    request_batch=True,
    request_shuffle=True,
    random_user_agent=False,
    adaptive_noise_budget=0.4,
    adaptive_delay_budget=0.1,
)
