from flask import Flask, request, jsonify
import numpy as np
import pandas as pd

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
    students = pd.json_normalize(request.json)

    main_criteria_score = students[['Home_Work_Score', 'Attendance_Score', 'Rate']].sum(axis=1)
    _, inv = np.unique(main_criteria_score, return_inverse=True)
    new_Elo = elo.get_new_ratings(students['Elo'].values, (inv + 1)[::-1].tolist())

    return jsonify({"ratings": new_Elo.tolist()})

if __name__ == '__main__':
    port = args.port
    host = args.host
    app.run(host=host, port=port, debug=True)