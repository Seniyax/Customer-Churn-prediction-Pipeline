
from flask import Flask,jsonify,request
import joblib

app = Flask(__name__)
model = joblib.load('C:\ Users\Dell\OneDrive\Desktop\ML\Customer Churn Pipeline\ Notebooks\model_final.pkl')

@app.route('/')
def home():
    return 'Model is running'

@app.route('/predict',methods=['POST'])
def predict():
    data = request.get_json(force=True)
    features = data['features']
    prediction = model.predict(features)
    return jsonify({'prediction': prediction})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
