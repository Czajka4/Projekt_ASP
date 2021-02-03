from flask import Flask, jsonify, make_response, Markup, render_template
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

@app.route('/chart/example')
def example():    
    return render_template('first_chart.htmls')

@app.route('/readjson')
def read_json():
    url = "http://api:5000/api/data/get_example"
    response = requests.get(url=url) #, json=jsonf)
    data = response.json()

    labels = list(data['x'])
    values = list(data['y'])

    return render_template('testchart.html', title='Sinus', max=1, 
                            labels=labels, values=values)





@app.errorhandler(404)
def not_found(error):
    return make_response(jsonify({'error': 'Not found'}), 404)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=port)