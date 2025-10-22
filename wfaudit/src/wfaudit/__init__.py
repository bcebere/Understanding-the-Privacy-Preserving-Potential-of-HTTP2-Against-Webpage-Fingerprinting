# stdlib
import sys

# wfaudit relative
from . import logger  # noqa: F401
from .benchmarks import audit  # noqa: F401
from .benchmarks import evaluate_leakage_from_deepse  # noqa: F401
from .benchmarks import evaluate_leakage_from_wefde  # noqa: F401
from .benchmarks import evaluate_ml_from_deepse  # noqa: F401
from .benchmarks import evaluate_ml_from_wefde  # noqa: F401
from .benchmarks import evaluate_shap  # noqa: F401
from .parser import _prepare_time_series_arrow  # noqa: F401
from .parser import merge_pcap_csvs  # noqa: F401
from .parser import prepare_all_datasets  # noqa: F401
from .parser import prepare_deepse_dataset  # noqa: F401; noqa: F401
from .parser import prepare_wefde_dataset  # noqa: F401; noqa: F401
from .parser import prepare_wefde_raw  # noqa: F401; noqa: F401
from .parser import process_raw_pcaps  # noqa: F401; noqa: F401

logger.add(sink=sys.stderr, level="DEBUG")
