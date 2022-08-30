# import firebase_admin 
# from firebase_admin import credentials
# from firebase_admin import db 
# import firebase

# cred = credentials.Certificate("./parking-monitoring-syste-19fda-firebase-adminsdk-4196p-f6342ad4bd.json")
# app=firebase_admin.initialize_app(cred, {'dbURL' : 'https://parking-monitoring-syste-19fda-default-rtdb.firebaseio.com/'})
# db=firebase.Database()
# print(db)
# dbase = db.Reference(path='/', app=app)
# ref=db.Reference(path='/',app=app)
# print(ref.get())
# users = dbase.child('users')
# data= {"abc":"abc"}
# dbase.push(data)
# dbase.child("users").set(data)
# # print(users.get())

# import pyrebase
# config={
#      "apiKey": "AIzaSyCuLaPPcybBGupc4TfqwR6Zlstxpk-LqXw",
#   "authDomain": "parking-monitoring-syste-19fda.firebaseapp.com",
#   "databaseURL": "https://parking-monitoring-syste-19fda-default-rtdb.firebaseio.com",
#   "projectId": "parking-monitoring-syste-19fda",
#   "storageBucket": "parking-monitoring-syste-19fda.appspot.com",
#   "messagingSenderId": "270432189432",
#   "appId": "1:270432189432:web:2e697a8b9bfb39f1c6f40d",
#   "measurementId": "G-P5XXE2QHV1"
# }
# firebase=pyrebase.initialize_app(config)
# db=firebase.database()

# data={"name":"dawood"}
# db.push(data)


from time import time
import firebase_admin
from firebase_admin import db

from datetime import date,timedelta
import datetime
import calendar
my_date = date.today()
pre_date=my_date-timedelta(days=1)
day = calendar.day_name[my_date.weekday()]
print(my_date)
print(calendar.day_name[my_date.weekday()]  )
time=datetime.datetime.now()
hour=time.strftime("%H")
print(hour)

cred_obj = firebase_admin.credentials.Certificate('./parking-monitoring-syste-19fda-firebase-adminsdk-4196p-f6342ad4bd.json')
databaseURL='https://parking-monitoring-syste-19fda-default-rtdb.firebaseio.com/'
default_app = firebase_admin.initialize_app(cred_obj, {
	'databaseURL':databaseURL
	})
ref = db.reference("Cars detected")

my_date=str(my_date)
day=str(day)
hour=str(hour)
# import json
# # Data to be written
# dictionary = {
#     my_date:
# 	{
# 		hour:5
# 	}
# }
 
# # Serializing json
# json_object = json.dumps(dictionary, indent=4)
 
# # Writing to sample.json
# with open("car_info.json", "w") as outfile:
#     outfile.write(json_object)
# with open("./car_info.json", "r") as f:
# 	file_contents = json.load(f)
# ref.set(file_contents)
# for i in range(0,10):
# 	ref.push({
# 	"date":my_date, "day":day,"hour":hour, "car_count":5
# })
outputData=ref.get()


# outputData1={}
Previous_Date = str(pre_date )
print("Previous_Date",Previous_Date)
cars_count=[]
for key,value in outputData.items():
    ref1 = db.reference("/Cars detected/"+key)
    o=(ref1.get())
    print("lll",o)
    
    # print(outputData)
    cars_count.append({"date":o["date"],"day":o["day"], "hour":o["hour"], "car_count":o["car_count"]})
    outputData1=(outputData[key])
print(cars_count,len(cars_count))
    
    # print("ddd",list(outputData1.keys())[0])
    # d=list(outputData1.keys())[0]
    # print(outputData1[d])
    # a=outputData1[d]
    # for keys in a:
    #     print("aaa",a[keys],len(a))
    #     print(a[keys]["count"])
    
    # for key1,value1 in outputData[key].items():
    #     print(outputData[key1])
    
    # cars_count.append(outputData[key])
    # print("value",value[0])
    # for i in value:
    #     print(value[i])
# print("Output data",outputData1)
# cars_count1=[]
# for i in cars_count:
#     # cars_count1.append(cars_count[i])
#     print(cars_count[i])
# print(cars_count1)

# for key, value in outputData.items():
#     print("key",outputData[key])
#     print("va;ue",value)
# Flask utils
from flask import Flask, redirect, url_for, request, render_template
from werkzeug.utils import secure_filename
from gevent.pywsgi import WSGIServer

# Define a flask app
app = Flask(__name__)
@app.route('/', methods=['GET','POST'])
def index():
    # Main page
    return render_template('index.html',cars_count=cars_count)

if __name__ == "__main__":
    app.run(debug=True)
