import requests

url = "http://127.0.0.1:8000/predict"
data = {
    "MedInc": 4.5,
    "HouseAge": 25,
    "Population": 3000,
    "AveOccup": 3.5,
    "Rooms_per_Household": 5.2,
    "Bedrooms_per_Room": 0.2,
    "Population_per_Household": 1.8
}

response = requests.post(url, json=data)

print(response.status_code, response.json())
