from .core_defense import DEFENSE

# Cherubin et al., 'Website Fingerprinting Defenses at the Application Layer', PETS 2017, Section 4.2.
# Delays every request by U(0, half the median page load time); tosses a coin
# on each request sent and each response received to issue an extra request.
# request_delay_max is overridden per dataset in levels.py.
LLAMA_DEFENSE = DEFENSE(
    name="llama",
    send_dummy_packet_strategy="llama",
    llama_dummy_probability=0.3,
    dummy_min_resource_size=0,
    dummy_on_response=True,
    request_delay=True,
    request_delay_probability=1.0,
    request_delay_max=1.5,
    request_shuffle=True,
    request_batch=True,
)
