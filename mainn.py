import logging
from pathlib import Path
from dotenv import load_dotenv

from livekit.agents import JobContext, WorkerOptions, cli
from livekit.agents.llm import function_tool
from livekit.agents.voice import Agent, AgentSession
from livekit.plugins import deepgram, openai, elevenlabs, silero

# ---------------------------------------------------------------------------
# Setup logging
# ---------------------------------------------------------------------------
logger = logging.getLogger("language-switcher")
logger.setLevel(logging.INFO)

# Load environment variables from .env file
load_dotenv(dotenv_path=Path(__file__).parent.parent / ".env")


class LanguageSwitcherAgent(Agent):
    """
    A voice-enabled assistant that can switch between multiple languages
    (English, Spanish, French, German, Italian, Tamil) when asked.
    """

    def __init__(self) -> None:
        super().__init__(
            instructions=(
                "You are a friendly voice assistant. "
                "You should speak clearly, and if the user asks, "
                "switch your responses into their chosen language. "
                "Avoid any unpronounceable characters."
            ),
            stt=deepgram.STT(model="nova-2-general", language="en"),
            llm=openai.LLM(model="gpt-4o"),
            tts=elevenlabs.TTS(model="eleven_turbo_v2_5", language="en"),
            vad=silero.VAD.load(),
        )

        # Default language is English
        self.current_language = "en"

        # Supported language mappings
        self.language_names = {
            "en": "English",
            "es": "Spanish",
            "fr": "French",
            "de": "German",
            "it": "Italian",
            "ta": "Tamil",
        }

        # Deepgram requires specific language codes
        self.deepgram_codes = {
            "en": "en",
            "es": "es",
            "fr": "fr-CA",  # using Canadian French as example
            "de": "de",
            "it": "it",
            "ta": "ta",  # Tamil language code for Deepgram
        }

        # Friendly greetings for each language
        self.greetings = {
            "en": "Hello! I'm now speaking in English. How can I help you today?",
            "es": "¡Hola! Ahora estoy hablando en español. ¿Cómo puedo ayudarte hoy?",
            "fr": "Bonjour ! Je parle maintenant en français. Comment puis-je vous aider aujourd'hui ?",
            "de": "Hallo! Ich spreche jetzt Deutsch. Wie kann ich Ihnen heute helfen?",
            "it": "Ciao! Ora sto parlando in italiano. Come posso aiutarti oggi?",
            "ta": "வணக்கம்! நான் இப்போது தமிழில் பேசுகிறேன். இன்று நான் உங்களுக்கு எவ்வாறு உதவ முடியும்?",
        }

    async def on_enter(self):
        """Called when the agent first joins the session."""
        await self.session.say(
            "Hi there! I can speak in English, Spanish, French, German, "
            "Italian, or Tamil. Just ask me to switch to one of them. "
            "Which language would you like me to use?"
        )

    async def _switch_language(self, code: str) -> None:
        """
        Switch the agent's STT and TTS systems to a new language.
        """
        if code == self.current_language:
            await self.session.say(
                f"I'm already speaking in {self.language_names[code]}."
            )
            return

        # Update speech-to-text
        if self.stt:
            new_code = self.deepgram_codes.get(code, code)
            self.stt.update_options(language=new_code)

        # Update text-to-speech
        if self.tts:
            self.tts.update_options(language=code)

        # Record the new current language
        self.current_language = code

        # Say a greeting in the new language
        await self.session.say(self.greetings[code])

    # -----------------------------------------------------------------------
    # Functions exposed to the LLM — each can be called to change language
    # -----------------------------------------------------------------------

    @function_tool
    async def switch_to_english(self):
        """Switch to English responses."""
        await self._switch_language("en")

    @function_tool
    async def switch_to_spanish(self):
        """Switch to Spanish responses."""
        await self._switch_language("es")

    @function_tool
    async def switch_to_french(self):
        """Switch to French responses."""
        await self._switch_language("fr")

    @function_tool
    async def switch_to_german(self):
        """Switch to German responses."""
        await self._switch_language("de")

    @function_tool
    async def switch_to_italian(self):
        """Switch to Italian responses."""
        await self._switch_language("it")

    @function_tool
    async def switch_to_tamil(self):
        """Switch to Tamil responses."""
        await self._switch_language("ta")


# ---------------------------------------------------------------------------
# Entrypoint for running inside LiveKit worker
# ---------------------------------------------------------------------------
async def entrypoint(ctx: JobContext):
    session = AgentSession()
    await session.start(agent=LanguageSwitcherAgent(), room=ctx.room)


if __name__ == "__main__":
    cli.run_app(WorkerOptions(entrypoint_fnc=entrypoint))