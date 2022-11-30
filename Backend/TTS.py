from gtts import gTTS
import os

mytext = "Test, test test Belgium is winning the wc"
audio = gTTS(text=mytext, lang="en", slow=False)
audio.save("example.mp3")
os.system("start example.mp3")


def textToSpeech(text, fileName):
    audio = gTTS(text=text, lang="en", slow=False)
    audio.save(fileName)


def playText(text, fileName):
    textToSpeech(text, fileName)
    os.system(f"start {fileName}")


playText(mytext, "example.mp3")
