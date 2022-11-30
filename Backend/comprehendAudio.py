import speech_recognition as sr
from os import path
import subprocess
from pydub import AudioSegment

r = sr.Recognizer()
test = sr.AudioFile("C:\\Users\\emiel\\OneDrive\\Documenten\\2ai\\xtopia-repo\\Backend\\output.wav")
with test as source:
    audio = r.record(source)

response = r.recognize_google(audio)

print(response)
import smtplib, ssl
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart


sender_email = "robot.xtopia@gmail.com"
receiver_email = "robot.xtopia@gmail.com"
password = "txotplydnakygjla"

message = MIMEMultipart("alternative")
message["Subject"] = "Vraagske"
message["From"] = sender_email
message["To"] = receiver_email

# Create the plain-text and HTML version of your messagedd
text = f"{response}"

# Turn these into plain/html MIMEText objects
part1 = MIMEText(text, "plain")

# Add HTML/plain-text parts to MIMEMultipart message
# The email client will try to render the last part first
message.attach(part1)

# Create secure connection with server and send email
context = ssl.create_default_context()
with smtplib.SMTP_SSL("smtp.gmail.com", 465, context=context) as server:
    server.login(sender_email, password)
    server.sendmail(
        sender_email, receiver_email, message.as_string())
