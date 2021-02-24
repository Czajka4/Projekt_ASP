from flask import Flask, jsonify, make_response, Markup, render_template, redirect, request, url_for
from datetime import datetime
from flask_cors import CORS
import platform
import os
from math import sin
import numpy as np
import requests


app = Flask(__name__)
CORS(app)
port = int(os.environ.get("PORT", 8000))
env = os.environ.get("FLASK_ENV", 'production')

@app.route('/api/info', methods=['GET'])
def get_info():
    hostname=platform.node()
    return jsonify({'hostname': hostname, 'port': str(port), 'env': str(env) })

@app.route("/")
def home():
    return redirect(url_for("stock"))

@app.route('/stock', methods=["POST", "GET"])
def stock():
    # default or first value is NVDA
    if request.method == "POST":
        symbol = str(request.form['ticker'])
        if symbol:
            symbol = symbol.upper()
        else:
            symbol = "NVDA"
    else:
         symbol = "NVDA"

    # ask api for data
    url = "http://api:5000/api/print/" + symbol
    response = requests.get(url=url)
    data = response.json()

    labels_val = list(data['values']['x'])
    values = list(data['values']['y'])

    labels_pred = list(data['predicted']['x'])
    predicted = list(data['predicted']['y'])

    maxv = int(float(max(values)) * 1.1)

    return render_template('testchart.html', title=" Kurs {} ".format(symbol), max=maxv, 
                            labels1=labels_val, values1=values,
                            labels2=labels_pred, values2=predicted)


@app.errorhandler(404)
def not_found(error):
    return make_response(jsonify({'error': 'Not found'}), 404)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=port) 