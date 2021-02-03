from flask import Flask, jsonify, make_response
from datetime import datetime
from flask_cors import CORS
import platform
import os
from math import sin
import numpy as np
import json


app = Flask(__name__)
CORS(app)
port = int(os.environ.get("PORT", 5000))
env = os.environ.get("FLASK_ENV", 'production')

@app.route('/api/info', methods=['GET'])
def get_info():
    hostname=platform.node()
    return jsonify({'hostname': hostname, 'port': str(port), 'env': str(env) })

@app.route('/api/date', methods=['GET'])
def get_date():
    now = datetime.now()
    year = now.strftime("%Y")
    month = now.strftime("%m")
    day = now.strftime("%d")
    time = now.strftime("%H:%M:")
    date_time = now.strftime("%m/%d/%Y, %H:%M:%S")
    return jsonify({'time': time, 'day': day, 'month': month, 'year':year, 'date_time': date_time})

@app.route("/api/data/get_example", methods=['GET'])
def get_example_data():
    x_val = np.round(np.linspace(0, 7, num=30, endpoint=False), decimals=3)
    y_val = np.round(np.array([sin(y) for y in x_val]), decimals=3)
    return jsonify({'x': x_val.tolist(), 'y':y_val.tolist()})



@app.errorhandler(404)
def not_found(error):
    return make_response(jsonify({'error': 'Not found'}), 404)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=port)