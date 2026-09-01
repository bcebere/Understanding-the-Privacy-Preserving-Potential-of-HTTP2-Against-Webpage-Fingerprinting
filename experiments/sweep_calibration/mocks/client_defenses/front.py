from .core_defense import DEFENSE

# Gong & Wang, USENIX Security 2020.
# Draws a per-connection padding window W and a dummy budget N, then emits
# dummies at Rayleigh(W) timestamps.  Zero delay: padding only.
#
# Deviations from the original, all downward:
#   - the original budget is ~1700 dummy PACKETS per direction; here N counts
#     dummy REQUESTS, each of which also pulls a response
#   - the original draws W from U(1, 14) s; a wider window here would be
#     truncated by max_dummy_time
#   - the original pads client and server; this is the client half only
FRONT_DEFENSE = DEFENSE(
    name="front",
    send_dummy_packet_strategy="front",
    send_dummy_packet_interval="front",
    send_dummy_packet_loop=True,
    send_dummy_min=1,
    send_dummy_max=200,
    front_window_min=0.0,
    front_window_max=1.0,
    max_dummy_time=2.0,
)
