from .core_defense import DEFENSE

# H2PC: the privacy-conscious HTTP/2 client of Section 6.1.1, Figure 8.
# Randomized receive window, request shuffling and batching, PING padding,
# and guarding noise streams around real requests.
H2PC_DEFENSE = DEFENSE(
    name="h2pc",
    initial_window_size_strategy="random",
    recv_interval_strategy=0.0001,
    recv_delay_threshold=10000,
    recv_threshold_resample=False,
    request_batch=True,
    request_shuffle=True,
    random_pings=True,
    ping_probability=0.5,
    ping_count_min=1,
    ping_count_max=3,
    send_dummy_packet_strategy="random_batch",
    send_dummy_packet_limit=1,
    send_dummy_packet_loop=False,
    max_dummy_time=2.0,
)
