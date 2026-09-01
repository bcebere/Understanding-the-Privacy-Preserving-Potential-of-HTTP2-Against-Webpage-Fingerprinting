from .core_defense import DEFENSE

# Cai et al., CCS 2014, as adapted to QUIC by Smith et al. (QCSD) and to
# HTTP/2 here.  Original Tamaraw sends fixed 750 B packets at rho_out = 0.02
# and rho_in = 0.006 s/packet, and pads the packet count to a multiple of
# L = 100.
#
# Mapping and deviations:
#   send_dummy_packet_interval  <- rho_out (0.02, as in the original)
#   recv_interval_strategy      <- rho_in  (0.01 here, 0.006 in the original)
#   initial_window_size_strategy    approximates the fixed packet size via
#                                   HTTP/2 flow control; frame-to-segment
#                                   mapping stays with the kernel
#   L has no counterpart: the trace is not padded to a multiple of a packet
#   count, so this variant lacks Tamaraw's anonymity-set guarantee
TAMARAW_QCSD_DEFENSE = DEFENSE(
    name="tamaraw_qcsd",
    initial_window_size_strategy=4096,
    recv_delay_threshold=4096,
    recv_interval_strategy=0.01,
    recv_threshold_resample=False,
    send_dummy_packet_strategy="random_per_connection",
    send_dummy_packet_interval=0.02,
    send_dummy_packet_loop=True,
    max_dummy_time=2.0,
)
