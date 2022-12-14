import torch
from IPython.display import Image, clear_output  # to display images

# #clear_output()
# print(f"Setup complete. Using torch {torch.__version__} ({torch.cuda.get_device_properties(0).name if torch.cuda.is_available() else 'CPU'})")
# print(torch.cuda.is_available())

#Firebase Libraries
import firebase_admin
from firebase_admin import db

def connectDB():
    if not firebase_admin._apps:
        cred_obj = firebase_admin.credentials.Certificate('./parking-monitoring-syste-19fda-firebase-adminsdk-4196p-f6342ad4bd.json')
        databaseURL='https://parking-monitoring-syste-19fda-default-rtdb.firebaseio.com/'
        default_app = firebase_admin.initialize_app(cred_obj, {
            'databaseURL':databaseURL
            })
    ref = db.reference("Cars detected")
    return ref

ref=connectDB()
    #print(model_predicted_count)
ref.push({
	"date":"2022-09-07", "day":"Wednesday","hour":"08", "car_count":"51"
})
print("data inserted")