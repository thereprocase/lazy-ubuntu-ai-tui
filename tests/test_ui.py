import asyncio

from textual.widgets import Select

from local_ai_tui.app import LocalAIApp
from local_ai_tui.core import GPU


def test_tui_mounts_and_changes_complexity(tmp_path, monkeypatch):
    monkeypatch.setenv("XDG_STATE_HOME", str(tmp_path))

    async def exercise():
        app = LocalAIApp()
        async with app.run_test() as pilot:
            await pilot.pause()
            assert app.query_one("#model-path")
            sample = [GPU(0, "GPU-test", "RTX A4000", 16384, "8.6")]
            app._detected(sample)
            await pilot.pause()
            app._detected(sample)
            await pilot.pause()
            assert app._selected_gpus() == ["GPU-test"]
            assert not app.query_one("#split-mode").display
            app.query_one("#complexity", Select).value = "expert"
            await pilot.pause()
            assert app.query_one("#split-mode").display

    asyncio.run(exercise())
