from pathlib import Path

from streamlit.testing.v1 import AppTest

APP = Path(__file__).resolve().parents[1] / "app" / "main.py"


def test_dashboard_renders_from_sample_data() -> None:
    at = AppTest.from_file(str(APP), default_timeout=60).run()
    assert not at.exception, at.exception
    assert [t.label for t in at.tabs] == ["Risk map", "Event feed", "Brief", "Backtest",
                                         "Forecast model"]  # fmt: skip
    assert any("Lanes exposed" in m.label for m in at.metric)
    assert any("Disruption brief" in md.value for md in at.markdown)
