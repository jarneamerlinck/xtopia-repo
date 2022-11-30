import pyaudio
import wave
import speech_recognition as sr
from os import path
import subprocess
from pydub import AudioSegment
import smtplib, ssl
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart


def RecordVoice(chunk=24, sample_format=pyaudio.paInt16, channels=1, fs=44100, seconds=3, filename="output.wav"):
    p = pyaudio.PyAudio()
    print('Recording')

    stream = p.open(format=sample_format,
                    channels=channels,
                    rate=fs,
                    frames_per_buffer=chunk,
                    input=True)

    frames = []  # Initialize array to store frames

    # Store data in chunks for 3 seconds
    for i in range(0, int(fs / chunk * seconds)):
        data = stream.read(chunk)
        frames.append(data)

    # Stop and close the stream
    stream.stop_stream()
    stream.close()
    # Terminate the PortAudio interface
    p.terminate()

    print('Finished recording')
    saveVoiceFile(p, frames)


def saveVoiceFile(p, frames, sample_format=pyaudio.paInt16, channels=1, fs=44100, seconds=3, filename="output.wav"):
    # Save the recorded data as a WAV file
    wf = wave.open(filename, 'wb')
    wf.setnchannels(channels)
    wf.setsampwidth(p.get_sample_size(sample_format))
    wf.setframerate(fs)
    wf.writeframes(b''.join(frames))
    wf.close()


test = sr.AudioFile("C:\\Users\\emiel\\OneDrive\\Documenten\\2ai\\xtopia-repo\\Backend\\output.wav")


def analyseVoiceFile(filePath):
    r = sr.Recognizer()
    test = sr.AudioFile(filePath)
    response = ""
    with test as source:
        audio = r.record(source)
    try:
        response = r.recognize_google(audio)
    except:
        print("Bericht niet begrepen, probeer opnieuw")

    print(response)


def send_mail(title, body, receiver):
    sender_email = "robot.xtopia@gmail.com"
    receiver_email = receiver
    password = "txotplydnakygjla"

    message = MIMEMultipart("alternative")
    message["Subject"] = title
    message["From"] = sender_email
    message["To"] = receiver_email

    text = f"{body}"
    part1 = MIMEText(text, "plain")
    message.attach(part1)

    context = ssl.create_default_context()
    with smtplib.SMTP_SSL("smtp.gmail.com", 465, context=context) as server:
        server.login(sender_email, password)
        server.sendmail(
            sender_email, receiver_email, message.as_string())
