from flask import Flask, request, jsonify

import argparse

from src.elo import Elo

parser = argparse.ArgumentParser(description='Choose option')
parser.add_argument('-p', '--port', type=int, default=8000)
parser.add_argument('-ht', '--host', type=str, default="0.0.0.0")
args = parser.parse_args()

app = Flask(__name__, template_folder='web', static_folder='web/public')
elo = Elo()

@app.post('/get_new_ratings')
def get_new_ratings():
    rating = request.json['ratings']
    order = request.json['orders']

    return jsonify({'ratings': elo.get_new_ratings(rating, order).tolist()})

if __name__ == '__main__':
    port = args.port
    host = args.host
    app.run(host=host, port=port, debug=True)