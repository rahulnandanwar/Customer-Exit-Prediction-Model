from flask import Flask, render_template, request
import numpy as np
import pickle

app = Flask(__name__, template_folder='Templates')

@app.route('/')
def Home():
    return render_template('Home.html')

@app.route('/Result', methods=['POST'])
def customerExitPrediction():
    filename = 'random_forest_model.pkl'
    rf_model = pickle.load(open(filename, 'rb'))
    sc = pickle.load(open('standard_scalar.pkl', 'rb'))
    if request.method == 'POST':
        int_features = [int(x) for x in request.form.values()]
        final_features = [np.array(int_features)]
        vect = sc.transform(final_features)
        my_prediction = rf_model.predict(vect)
    return render_template('Result.html', prediction = my_prediction)

if __name__ == "__main__":
    app.run(debug = True)


