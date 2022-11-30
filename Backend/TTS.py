from gtts import gTTS
import os
def textToSpeech(text, fileName):
    audio = gTTS(text=text, lang="en", slow=False)
    audio.save(fileName)


def playText(text, fileName):
    textToSpeech(text, fileName)
    os.system(f"start {fileName}")


playText("mytext", "example.mp3")
