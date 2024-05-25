import pickle
from flask import Flask, request, jsonify
from perceptron import load_perceptron

app = Flask(__name__)

class CustomUnpickler(pickle.Unpickler):
    def find_class(self, module, name):
        if name == 'Perceptron':
            return load_perceptron()
        return super().find_class(module, name)

@app.route('/predict_get', methods=['GET'])
def get_prediction():
    sepal_length = float(request.args.get('sl'))
    petal_length = float(request.args.get('pl'))
    features = [sepal_length, petal_length]

    with open('model.pkl', "rb") as picklefile:
        model = CustomUnpickler(picklefile).load()

    predicted_class = int(model.predict(features))
    return jsonify(features=features, predicted_class=predicted_class)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
