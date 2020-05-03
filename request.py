import requests

url = 'http://localhost:5000/predict_api'
requests.post(url,json={'tage':20, 'tdiabetes':'YES', 'tasthma':'YES', 'theart':'NO'})

#print(r.json())