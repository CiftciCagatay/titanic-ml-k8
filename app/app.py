from flask import Flask, jsonify, request
import pickle

MODEL_FILE_PATH = 'model.pkl'
PORT = 8080

app = Flask(__name__)

@app.route('/predict', methods=['GET'])
def predict():
    age = request.args.get('age', default=0, type=int)
    prediction = model.predict([[age]])
    return jsonify({'prediction': list(prediction)})


if __name__ == '__main__':
    model = pickle.load(open(MODEL_FILE_PATH, 'rb'))
    app.run(host='0.0.0.0', port=PORT)
