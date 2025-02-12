# stdlib
import sys

# wfaudit relative
from . import logger  # noqa: F401
from .benchmarks import evaluate_all  # noqa: F401
from .benchmarks import evaluate_exploratory  # noqa: F401; noqa: F401
from .benchmarks import evaluate_exploratory_ml  # noqa: F401; noqa: F401
from .benchmarks import evaluate_leakage  # noqa: F401
from .benchmarks import evaluate_leakage_v2  # noqa: F401
from .benchmarks import evaluate_ml  # noqa: F401
from .benchmarks import prepare_features  # noqa: F401; noqa: F401
from .parser import create_datasets  # noqa: F401
from .parser import merge_pcap_csvs  # noqa: F401
from .parser import prepare_ts_datasets  # noqa: F401
from .parser import process_raw_pcaps  # noqa: F401; noqa: F401

logger.add(sink=sys.stderr, level="INFO")
