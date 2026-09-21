import json
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


@pytest.fixture
def saved_state():
    """Read the saved run metadata and input records for assertions after a run."""

    def read(output):
        metadata = output / ".flowde"
        run_metadata = json.loads(
            (metadata / "run_metadata.state").read_text(encoding="utf-8")
        )
        records = [
            json.loads(path.read_text(encoding="utf-8"))
            for path in (metadata / "input_records").glob("*.state")
        ]
        return {
            **run_metadata,
            "input_records": {record["key"]: record for record in records},
        }

    return read
