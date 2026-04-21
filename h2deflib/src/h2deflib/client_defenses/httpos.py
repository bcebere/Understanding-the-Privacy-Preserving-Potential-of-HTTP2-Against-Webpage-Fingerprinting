# wfaudit relative
# h2deflib relative
from .core_defense import DEFENSE

HTTPOS_DEFENSE = DEFENSE(
    name="httpos",
    ranged_requests=True,
)
