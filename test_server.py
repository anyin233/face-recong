import requests

SERVER_ADDR = "http://182.92.170.212:5000/predict_res"

response = requests.post(SERVER_ADDR,
                         files={"file": open(
                             "data/faces/30601258@N03/coarse_tilt_aligned_face.2.8623020082_578acef81b_o.jpg", "rb")})
print(response)
print(response.json())
