import numpy as np
from flask import Flask, redirect, render_template, request, app
import paho.mqtt.client as mqtt

app = Flask(__name__)


@app.route('/', methods=['GET', 'POST'])
def index():
    return 'hello world'


if __name__ == '__main__':
    app.run()
