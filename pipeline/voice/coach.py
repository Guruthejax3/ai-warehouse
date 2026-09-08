"""Voice coach — optional spoken coaching messages behind a feature flag.

Priority: lowest. By default this module is a clean interface with a
no-op implementation; text is logged when VOICE_COACHING_ENABLED = false.

Two engines are supported when enabled:
    - "pyttsx3": offline TTS, works with no API key.
    - "gtts":    Google TTS, needs network. Lazy-imported.

The pipeline never blocks on a voice failure: the coach catches everything
and downgrades to a log line.
"""

from __future__ import annotations

import logging
import os

logger = logging.getLogger(__name__)


class VoiceCoach:
    """Generate and speak coaching messages for scored risk events.

    Args:
        enabled: overall switch (defaults to env VOICE_COACHING_ENABLED).
        engine: "pyttsx3" (default) or "gtts".
        rate: speech rate when the engine supports it.
    """

    def __init__(
        self,
        enabled: bool = False,
        engine: str = "pyttsx3",
        rate: int = 150,
    ) -> None:
        env_enabled = os.environ.get("VOICE_COACHING_ENABLED", "0") == "1"
        self.enabled = enabled or env_enabled
        self.engine = engine
        self.rate = rate
        self._tts = None

    # ------------------------------------------------------------------ public

    def build_message(self, event) -> str:
        """Compose a short, actionable coaching message from a RiskEvent."""
        level = getattr(event, "risk_level", "low")
        cls = getattr(event, "behavior_class", "unknown behaviour")
        is_blocked = bool(
            getattr(event, "metadata", {}) and event.metadata.get("gated_by_physics", False)
        )
        if is_blocked:
            return (
                f"Review only: {cls} detected but physics did not confirm it. "
                "No alert raised."
            )
        if level == "critical" or level == "high":
            return (
                f"High risk behaviour: {cls}. Use approved handling technique "
                "and required equipment immediately."
            )
        if level == "medium":
            return (
                f"Medium risk behaviour: {cls}. Please correct your technique."
            )
        return f"Behaviour noted: {cls}. Continue safe handling."

    def speak(self, text: str) -> bool:
        """Speak (or log) a coaching message. Never raises.

        Returns True if audio was actually produced.
        """
        if not self.enabled:
            logger.info("[voice disabled] %s", text)
            return False
        try:
            self._ensure_tts()
            if self._tts is None:
                logger.warning("[voice] engine %r unavailable — logging only.", self.engine)
                logger.info("[voice] %s", text)
                return False
            self._speak_impl(text)
            return True
        except Exception as exc:  # never let voice break the pipeline
            logger.warning("[voice] failed (%s) — logging only.", exc)
            logger.info("[voice] %s", text)
            return False

    def coach(self, event) -> bool:
        """One-shot: build + speak a message for an event."""
        return self.speak(self.build_message(event))

    # ------------------------------------------------------------------ impl

    def _ensure_tts(self) -> None:
        if self._tts is not None:
            return
        if self.engine == "pyttsx3":
            try:
                import pyttsx3  # type: ignore

                self._tts = pyttsx3.init()
                self._tts.setProperty("rate", self.rate)
            except Exception as exc:
                logger.debug("pyttsx3 unavailable: %s", exc)
                self._tts = None
        elif self.engine == "gtts":
            # Keep the engine object as a flag; gTTS writes a temp file.
            try:
                from gtts import gTTS  # type: ignore

                self._tts = gTTS  # type: ignore
            except Exception as exc:
                logger.debug("gtts unavailable: %s", exc)
                self._tts = None
        else:
            self._tts = None

    def _speak_impl(self, text: str) -> None:
        if self.engine == "pyttsx3" and self._tts is not None:
            self._tts.say(text)
            self._tts.runAndWait()
        elif self.engine == "gtts" and self._tts is not None:
            import tempfile

            import playsound  # type: ignore

            cls = self._tts  # the gTTS class
            tts = cls(text=text)
            with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as f:
                name = f.name
            tts.save(name)
            playsound.playsound(name)
            import os

            os.remove(name)