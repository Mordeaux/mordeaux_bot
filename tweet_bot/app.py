#!/usr/bin/python2
"""
This is the entry point for the Twitter bot web application.
Author: Michael Mordowanec (mordeaux)
"""


from flask import Flask, render_template


app = Flask(__name__)


@app.route('/')
def home():
    """
    This route returns the Single Page Application.
    """
    return render_template('index.html')
