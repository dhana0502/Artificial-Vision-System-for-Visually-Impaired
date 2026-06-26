import threading
import subprocess
import tempfile
import os

_lock = threading.Lock()
_last_spoken = ""
_speaking = False


def speak(text):
    """
    Speaks the given text using Windows' built-in PowerShell speech
    synthesizer (System.Speech). Writes a temporary .ps1 script and
    runs it using the full path to powershell.exe.
    """
    global _last_spoken, _speaking

    if not text or text == _last_spoken or _speaking:
        return

    _last_spoken = text

    def _run():
        global _speaking
        _speaking = True
        script_path = os.path.join(tempfile.gettempdir(), "av_speak.ps1")
        try:
            with _lock:
                safe_text = text.replace("'", "''")
                script_content = (
                    "Add-Type -AssemblyName System.speech\n"
                    "$speak = New-Object System.Speech.Synthesis.SpeechSynthesizer\n"
                    f"$speak.Speak('{safe_text}')\n"
                )
                with open(script_path, "w", encoding="utf-8") as f:
                    f.write(script_content)

                powershell_path = os.path.join(
                    os.environ.get("SystemRoot", r"C:\Windows"),
                    "System32", "WindowsPowerShell", "v1.0", "powershell.exe"
                )

                result = subprocess.run(
                    [powershell_path, "-ExecutionPolicy", "Bypass", "-File", script_path],
                    capture_output=True,
                    text=True,
                    timeout=20,
                )
                if result.returncode != 0:
                    print("TTS PowerShell error:")
                    print(result.stderr)
        except Exception as e:
            print(f"TTS error: {e}")
        finally:
            _speaking = False

    threading.Thread(target=_run, daemon=True).start()


def describe_face(face_features):
    """
    Builds a natural sentence from face_features dict and speaks it.
    """
    name = face_features.get("name", "unknown")
    age = face_features.get("age", "N/A")
    gender = face_features.get("gender", "N/A")
    emotion = face_features.get("emotion", "N/A")

    if name and name != "unknown":
        sentence = f"{name} is nearby, looking {emotion}."
    else:
        sentence = f"An unknown {gender} person, around {age} years old, looking {emotion}."

    speak(sentence)