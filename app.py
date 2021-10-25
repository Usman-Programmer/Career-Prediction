from flask import Flask, request, jsonify
import numpy as np
import pickle
model=pickle.load(open('model.pkl','rb'))

app = Flask(__name__)

@app.route('/')
def index():
    return "Hello world"

@app.route('/predict', methods=['GET', 'POST'])
def predict():
    METRIC = request.form.get('METRIC')
    INTER= request.form.get('INTER')
    NTS= request.form.get('NTS')
    DEPARTMENT=request.form.get('DEPARTMENT')
    INSTITUTE=request.form.get('INSTITUTE')

    input_query=np.array([[METRIC,INTER,DEPARTMENT,NTS,INSTITUTE]])
    result=model.predict(input_query)[0]
    return jsonify({'ADMISSION' : str(result)})

if __name__ == '__main__':
    app.run(debug=True)