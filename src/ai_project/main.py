import sys
import os

# Add the project root (parent of both folders) to the path
sys.path.append(os.path.join(os.path.dirname(__file__), "..", "..", "Ai_interface"))

from voicepage import DeepgramVoiceAssistantGUI
def main():
    app = DeepgramVoiceAssistantGUI()
    app.mainloop()


if __name__ == "__main__":
    main()