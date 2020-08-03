import requests

url = 'http://localhost:5000/predict_api'
r = requests.post(url,json={'age':40, 'healthy_eating':9, 'active_lifestyle':6,'blood_group': 1})

print(r.json())


