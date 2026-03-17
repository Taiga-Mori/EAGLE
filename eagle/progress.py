class ConsoleProgress:
    """Minimal console progress reporter compatible with Streamlit's progress API."""

    def __init__(self) -> None:
        self._last_text = None
        self._last_percent = None

    def progress(self, value: float, text: str = "") -> None:
        percent = int(round(max(0.0, min(1.0, value)) * 100))
        if text == self._last_text and percent == self._last_percent:
            return
        self._last_text = text
        self._last_percent = percent
        print(f"[{percent:3d}%] {text}", flush=True)
