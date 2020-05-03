import numpy as np
from flask import Flask, request, jsonify, render_template
import pickle

app = Flask(__name__)
model = pickle.load(open('model.pkl', 'rb'))

@app.route('/')
def home():
    return render_template('model.html')

@app.route('/predict',methods=['POST'])
def predict():
    '''
    For rendering results on HTML GUI
    '''
#    for n in request.form.values():
#        if (n != "YES"):
#            return render_template('error in input')
        
    
    int_features = [int(x) for x in request.form.values()]
    final_features = [np.array(int_features)]
    #return render_template('model.html', prediction_text='severity % rate could be {}%'.format(n))
    
    prediction = model.predict(final_features)

    output = int(round(prediction[0], 2)) 
    if (output > 99):
        output = 99
    return render_template('model.html', prediction_text='Severity rate could be {}%'.format(output))

@app.route('/predict_api',methods=['POST'])
def predict_api():
    '''
    For direct API calls trought request
    '''
    data = request.get_json(force=True)
    prediction = model.predict([np.array(list(data.values()))])

    output = prediction[0]
    return jsonify(output)

def page_not_found(e):
  return render_template('404.html'), 404

if __name__ == "__main__":
    app.run(debug=True)