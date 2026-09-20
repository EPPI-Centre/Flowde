from multiprocessing.managers import SyncManager

import pytest
from joblib.externals.loky.backend.context import get_context


@pytest.fixture(scope="session")
def process_manager():
    with SyncManager(ctx=get_context()) as manager:
        yield manager


@pytest.fixture
def calls(process_manager):
    """Observe calls across the image worker process boundary."""
    return process_manager.list()
