# stdlib
import sys

# wfaudit relative
from . import logger  # noqa: F401
from .parser import (  # noqa: F401
    create_datasets,
    merge_pcap_csvs,
    prepare_wefde_datasets,
    process_raw_pcaps,
)

logger.add(sink=sys.stderr, level="DEBUG")
