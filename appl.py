from flask import Flask, request, jsonify, render_template
import pickle

appl = Flask(__name__)

# Load the trained model
model_path = r'/model/model.pkl'  # Update the path if necessary
with open(model_path, 'rb') as model_file:
    model = pickle.load(model_file)

@appl.route('/')
def index():
    return render_template('message.html')

@appl.route('/predict', methods=['POST'])
def predict():
    data = request.get_json(force=True)
    user_input = data['input']

    # Generate a prediction using the loaded model
    prediction = model.predict([user_input])[0]

    return jsonify({"prediction": [prediction]})

if __name__ == '__main__':
    appl.run(host='0.0.0.0', port=5000, debug=True)
