from flask import Flask, request, jsonify, render_template
import joblib


app = Flask(__name__)

# Load the trained model
model_file = 'model/model.pkl'
vect_file = 'model/vect.pkl'
  # Update with your path if different

model = joblib.load(model_file)
vect = joblib.load(vect_file)

def find_response(user_input):
    user_input = user_input.lower()
    model['user_input_lower'] = model['user_input'].str.lower()

    # Look for an exact or partial match in user inputs
    for idx, row in model.iterrows():
        if user_input in row['user_input_lower']:
            return row['bot_respond']

@app.route('/')
def index():
    return render_template('message.html')

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json(force=True)
    user_input = data['input']

    print('user_input')
    print(user_input)

    # Generate a prediction using the loaded model
    # Assuming the input is already preprocessed as needed by your model

    inp = vect.transform([user_input])

    prediction = model.predict(inp)[0]

    print('prediction')
    print(prediction)

    return jsonify({"prediction": [prediction]})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
