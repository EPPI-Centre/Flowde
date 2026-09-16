import pytest

from flowde.api_utils.openai_utils import check_openai_connection

pytestmark = pytest.mark.integration


@pytest.mark.parametrize("from_azure", [False, True], ids=["openai", "azure"])
def test_check_openai_connection_accepts_live_credentials(from_azure):
    assert (
        check_openai_connection(
            model="gpt-5.6-luna",
            effort="low",
            from_azure=from_azure,
        )
        is None
    )
