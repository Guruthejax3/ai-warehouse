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

#: Behaviour-class display names for the multilingual voice coach.
_BEHAVIOR_LABELS = {
    "en": {
        "product_dropped": "a product was dropped",
        "product_dragged": "a product was dragged",
        "rough_handling": "rough handling",
        "incorrect_stacking": "incorrect stacking",
        "unstable_stacking": "unstable stacking",
        "outside_designated_zone": "material outside the designated zone",
        "no_required_equipment": "missing required equipment",
        "pallet_mispositioned": "a pallet is mispositioned",
        "material_pushed_or_thrown": "material was pushed or thrown",
        "unsafe_loading_sequence": "an unsafe loading sequence",
        "forklift_pedestrian_conflict": "a forklift pedestrian near miss",
        "unknown behaviour": "unidentified behaviour",
    },
    "hi": {
        "product_dropped": "उत्पाद गिर गया",
        "product_dragged": "उत्पाद घसीटा गया",
        "rough_handling": "लापरवाही से संभालना",
        "incorrect_stacking": "गलत तरीके से रैकिंग",
        "unstable_stacking": "अस्थिर रैकिंग",
        "outside_designated_zone": "सामग्री निर्धारित क्षेत्र से बाहर",
        "no_required_equipment": "आवश्यक उपकरण अनुपस्थित",
        "pallet_mispositioned": "पैलेट गलत स्थिति में है",
        "material_pushed_or_thrown": "सामग्री धकेली या फेंकी गई",
        "unsafe_loading_sequence": "असुरक्षित लोडिंग अनुक्रम",
        "forklift_pedestrian_conflict": "फोर्कलिफ्ट पैदल यात्री निकट दुर्घटना",
        "unknown behaviour": "अज्ञात व्यवहार",
    },
    "te": {
        "product_dropped": "ఉత్పత్తి పడిపోయింది",
        "product_dragged": "ఉత్పత్తి లాగబడింది",
        "rough_handling": "నిర్లక్ష్యంగా నిర్వహించడం",
        "incorrect_stacking": "తప్పుగా పేర్చడం",
        "unstable_stacking": "అస్థిరమైన పేర్చడం",
        "outside_designated_zone": "పదార్థం నిర్దేశిత ప్రాంతం వెలుపల",
        "no_required_equipment": "అవసరమైన పరికరాలు లేవు",
        "pallet_mispositioned": "ప్యాలెట్ తప్పు స్థానంలో ఉంది",
        "material_pushed_or_thrown": "పదార్థం నెట్టబడింది లేదా విసిరివేయబడింది",
        "unsafe_loading_sequence": "అసురక్షిత లోడింగ్ క్రమం",
        "forklift_pedestrian_conflict": "ఫోర్క్‌లిఫ్ట్ పాదచారి సమీప ఘర్షణ",
        "unknown behaviour": "తెలియని ప్రవర్తన",
    },
    "es": {
        "product_dropped": "un producto se cayó",
        "product_dragged": "un producto fue arrastrado",
        "rough_handling": "manipulación brusca",
        "incorrect_stacking": "apilado incorrecto",
        "unstable_stacking": "apilado inestable",
        "outside_designated_zone": "material fuera de la zona designada",
        "no_required_equipment": "falta el equipo requerido",
        "pallet_mispositioned": "palé mal posicionado",
        "material_pushed_or_thrown": "material empujado o lanzado",
        "unsafe_loading_sequence": "secuencia de carga insegura",
        "forklift_pedestrian_conflict": "encuentro cercano entre montacargas y peatón",
        "unknown behaviour": "comportamiento no identificado",
    },
}

#: High/medium/low message frames per language. {cls} is substituted with the
#: translated behaviour label.
_MESSAGE_FRAMES = {
    "en": {
        "blocked": "Review only: {cls} was detected, but physics did not confirm it. No alert raised.",
        "high": "High risk behaviour: {cls}. Use the approved handling technique and required equipment immediately.",
        "medium": "Medium risk behaviour: {cls}. Please correct your technique.",
        "low": "Behaviour noted: {cls}. Continue safe handling.",
    },
    "hi": {
        "blocked": "केवल समीक्षा: {cls} पाया गया, परंतु भौतिकी ने इसकी पुष्टि नहीं की। कोई अलर्ट जारी नहीं।",
        "high": "उच्च जोखिम व्यवहार: {cls}। तुरंत स्वीकृत तकनीक और आवश्यक उपकरण का उपयोग करें।",
        "medium": "मध्यम जोखिम व्यवहार: {cls}। कृपया अपनी तकनीक सुधारें।",
        "low": "व्यवहार नोट किया गया: {cls}। सुरक्षित संचालन जारी रखें।",
    },
    "te": {
        "blocked": "సమీక్ష మాత్రమే: {cls} గుర్తించబడింది, కానీ భౌతిక శాస్త్రం దీనిని ధృవీకరించలేదు. అలర్ట్ లేదు.",
        "high": "అధిక ప్రమాద ప్రవర్తన: {cls}. వెంటనే ఆమోదించబడిన పద్ధతి మరియు అవసరమైన పరికరాలను ఉపయోగించండి.",
        "medium": "మధ్యస్థ ప్రమాద ప్రవర్తన: {cls}. దయచేసి మీ పద్ధతిని సరిచేయండి.",
        "low": "ప్రవర్తన గుర్తించబడింది: {cls}. సురక్షిత నిర్వహణ కొనసాగించండి.",
    },
    "es": {
        "blocked": "Solo revisión: se detectó {cls}, pero la física no lo confirmó. No se emitió ninguna alerta.",
        "high": "Comportamiento de alto riesgo: {cls}. Use la técnica aprobada y el equipo requerido inmediatamente.",
        "medium": "Comportamiento de riesgo medio: {cls}. Corrija su técnica.",
        "low": "Comportamiento anotado: {cls}. Continúe con el manejo seguro.",
    },
}


class VoiceCoach:
    """Generate and speak coaching messages for scored risk events.

    Args:
        enabled: overall switch (defaults to env VOICE_COACHING_ENABLED).
        engine: "pyttsx3" (default) or "gtts".
        rate: speech rate when the engine supports it.
        language: default language for alerts (en | hi | te | es).
    """

    def __init__(
        self,
        enabled: bool = False,
        engine: str = "pyttsx3",
        rate: int = 150,
        language: str = "en",
    ) -> None:
        env_enabled = os.environ.get("VOICE_COACHING_ENABLED", "0") == "1"
        self.enabled = enabled or env_enabled
        self.engine = engine
        self.rate = rate
        self.language = language if language in _MESSAGE_FRAMES else "en"
        self._tts = None

    # ------------------------------------------------------------------ public

    def build_message(self, event, language: Optional[str] = None) -> str:
        """Compose a short, actionable coaching message from a RiskEvent.

        ``language`` overrides the coach default (en | hi | te | es). gTTS
        speaks the returned text in that language; pyttsx3 reads the translated
        message with its default voice.
        """
        lang = (language or self.language)
        if lang not in _MESSAGE_FRAMES:
            lang = "en"
        frames = _MESSAGE_FRAMES[lang]
        labels = _BEHAVIOR_LABELS.get(lang, _BEHAVIOR_LABELS["en"])

        level = getattr(event, "risk_level", "low")
        cls = getattr(event, "behavior_class", "unknown behaviour")
        cls_label = labels.get(cls, cls.replace("_", " "))
        is_blocked = bool(
            getattr(event, "metadata", {}) and event.metadata.get("gated_by_physics", False)
        )
        if is_blocked:
            return frames["blocked"].format(cls=cls_label)
        if level == "critical" or level == "high":
            return frames["high"].format(cls=cls_label)
        if level == "medium":
            return frames["medium"].format(cls=cls_label)
        return frames["low"].format(cls=cls_label)

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