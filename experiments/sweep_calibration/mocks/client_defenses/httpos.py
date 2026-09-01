from .core_defense import DEFENSE

# Luo et al., 'HTTPOS: Sealing Information Leaks with Browser-side Obfuscation of Encrypted Flows', NDSS 2011.
# Four mechanisms: byte-range splitting, request-size
# padding via HTTP headers, receive-window manipulation, and pipelining.

HTTPOS_DEFENSE = DEFENSE(
    name="httpos",
    ranged_requests=True,
    send_packet_size_strategy="random_per_frame",
    initial_window_size_strategy=2048,
)
