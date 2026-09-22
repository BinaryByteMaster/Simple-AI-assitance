import os
import json
import queue
import threading
from dotenv import load_dotenv

from deepgram import DeepgramClient
from deepgram.core.events import EventType
from deepgram.agent.v1.types import (
    AgentV1Settings,
    AgentV1SettingsAgent,
    AgentV1SettingsAgentListen,
    AgentV1SettingsAgentListenProvider_V2,
    AgentV1SettingsAudio,
    AgentV1SettingsAudioInput,
    AgentV1SettingsAudioOutput,
)
from deepgram.types.think_settings_v1 import ThinkSettingsV1
from deepgram.types.think_settings_v1provider import ThinkSettingsV1Provider_OpenAi
from deepgram.types.speak_settings_v1 import SpeakSettingsV1
from deepgram.types.speak_settings_v1provider import SpeakSettingsV1Provider_Deepgram

load_dotenv()
DEEPGRAM_API_KEY = os.getenv("DEEPGRAM_API_KEY")

# Absolute path so it doesn't matter what directory the script is launched from
AI_JSON_PATH = os.path.join(os.path.dirname(__file__), "..", "Ai.json")


def load_agent_settings(path: str = AI_JSON_PATH) -> AgentV1Settings:
    """Load Ai.json and translate it into the SDK's typed settings object."""
    with open(path, "r", encoding="utf-8") as f:
        raw = json.load(f)

    audio_cfg = raw["audio"]
    agent_cfg = raw["agent"]

    return AgentV1Settings(
        audio=AgentV1SettingsAudio(
            input=AgentV1SettingsAudioInput(
                encoding=audio_cfg["input"]["encoding"],
                sample_rate=audio_cfg["input"]["sample_rate"],
            ),
            output=AgentV1SettingsAudioOutput(
                encoding=audio_cfg["output"]["encoding"],
                sample_rate=audio_cfg["output"]["sample_rate"],
                container=audio_cfg["output"]["container"],
            ),
        ),
        agent=AgentV1SettingsAgent(
            listen=AgentV1SettingsAgentListen(
                provider=AgentV1SettingsAgentListenProvider_V2(
                    type=agent_cfg["listen"]["provider"]["type"],
                    model=agent_cfg["listen"]["provider"]["model"],
                )
            ),
            think=ThinkSettingsV1(
                provider=ThinkSettingsV1Provider_OpenAi(
                    type=agent_cfg["think"]["provider"]["type"],
                    model=agent_cfg["think"]["provider"]["model"],
                ),
                prompt=agent_cfg["think"]["prompt"],
            ),
            speak=SpeakSettingsV1(
                provider=SpeakSettingsV1Provider_Deepgram(
                    type=agent_cfg["speak"]["provider"]["type"],
                    model=agent_cfg["speak"]["provider"]["model"],
                )
            ),
            greeting=agent_cfg.get("greeting"),
        ),
    )


def _handle_agent_message(msg_queue, msg):
    """Translate Deepgram Voice Agent events into (kind, payload) for the GUI queue."""
    if isinstance(msg, bytes):
        msg_queue.put(("audio", msg))
        return 
    msg_type = getattr(msg, "type", None)

    if msg_type == "ConversationText":
        role = msg.role
        content = msg.content
        msg_queue.put(("message", (role, content)))

    elif msg_type == "UserStartedSpeaking":
        msg_queue.put(("state", "listening"))

    elif msg_type == "AgentThinking":
        msg_queue.put(("state", "processing"))
        msg_queue.put(("interim", ("assistant", msg.content)))

    elif msg_type == "AgentStartedSpeaking":
        msg_queue.put(("state", "speaking"))

    elif msg_type == "AgentAudioDone":
        msg_queue.put(("state", "idle"))

    elif msg_type == "Welcome":
        msg_queue.put(("log", ("system", "Connected to Deepgram agent.")))

    elif msg_type == "SettingsApplied":
        msg_queue.put(("log", ("system", "Settings applied.")))

    elif msg_type == "Error":
        msg_queue.put(("log", ("system", f"Error [{msg.code}]: {msg.description}")))

    elif msg_type == "Warning":
        msg_queue.put(("log", ("system", f"Warning [{msg.code}]: {msg.description}")))

    elif msg_type == "History":
        pass  # ignore replay for now

    elif msg_type == "LatencyReport":
        # Ignore latency reports during normal operation
        pass

    else:
        msg_queue.put(("log", ("system", f"[unhandled event] {msg_type}: {msg}")))

def start_agent_connection(msg_queue):
    client = DeepgramClient(api_key=DEEPGRAM_API_KEY)
    settings = load_agent_settings(AI_JSON_PATH)

    # Wait until Deepgram confirms that settings were applied
    settings_applied = threading.Event()

    connection_cm = client.agent.v1.connect()
    connection = connection_cm.__enter__()

    connection.on(
        EventType.OPEN,
        lambda _: msg_queue.put(
            ("log", ("system", "Connection opened."))
        )
    )

    def handle_message(msg):
        # Detect SettingsApplied
        if getattr(msg, "type", None) == "SettingsApplied":
            settings_applied.set()

        # Send message to GUI handler
        _handle_agent_message(msg_queue, msg)

    connection.on(EventType.MESSAGE, handle_message)

    connection.on(
        EventType.CLOSE,
        lambda _: msg_queue.put(
            ("log", ("system", "Connection closed."))
        )
    )

    connection.on(
        EventType.ERROR,
        lambda err: msg_queue.put(
            ("log", ("system", f"Connection error: {err}"))
        )
    )

    # Send settings to Deepgram
    connection.send_settings(settings)

    # Start listening for Deepgram responses
    threading.Thread(
        target=connection.start_listening,
        daemon=True
    ).start()

    # Wait for Deepgram to confirm settings
    if not settings_applied.wait(timeout=5):
        msg_queue.put(
            (
                "log",
                (
                    "system",
                    "Timed out waiting for Deepgram SettingsApplied."
                )
            )
        )

    return connection, connection_cm


def stop_agent_connection(connection_cm):
    """Call this when the user stops the assistant / closes the app."""
    connection_cm.__exit__(None, None, None)


def send_audio_chunk(connection, audio_bytes: bytes):
    if connection is None:
        return False

    try:
        connection.send_media(audio_bytes)
        return True

    except Exception as e:
        print(f"Deepgram send_media error: {e}")
        return False