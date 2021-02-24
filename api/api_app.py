from flask import Flask, jsonify, make_response
from datetime import datetime
from flask_cors import CORS
import platform
import os
from math import sin
import numpy as np
import json
import pandas as pd

from scraping import scrape_yahoo
from arima_model import calculate_arima

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

@app.route("/api/print/", defaults={'symbol' : 'FB'}, methods=['GET'])
@app.route("/api/print/<string:symbol>")
def get_yahoo_data(symbol):
    close_val = scrape_yahoo(symbol)
    close_val, predict_val = calculate_arima(close_val)

    xc = np.array(pd.to_datetime(close_val.index.values).strftime('%Y.%m.%d'))
    yc = np.array(close_val.price.iloc[:])
    xp = np.array(pd.to_datetime(predict_val.index.values).strftime('%Y.%m.%d'))
    yp = np.array(predict_val.price.iloc[:])
        
    return jsonify({"values": {'x': xc.tolist(), 'y': yc.tolist()}, 
                    "predicted": {'x': xp.tolist(), 'y': yp.tolist() } } )

@app.errorhandler(404)
def not_found(error):
    return make_response(jsonify({'error': 'Not found'}), 404)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=port)