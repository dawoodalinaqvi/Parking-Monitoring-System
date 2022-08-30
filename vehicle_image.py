from flask import Flask, render_template
import os

moving_vehicle = os.path.join('static', 'moving_vehicle')

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = moving_vehicle

moved_vehicle = [1,5]
vehicle_paths=[]
@app.route('/')
# @app.route('/index')
def show_index():
    for i in moved_vehicle:
        full_filename = os.path.join(app.config['UPLOAD_FOLDER'], str(i)+'.jpg')
        full_filename.split('\\')
        vehicle_paths.append({"path":full_filename,"vehicle_number":i})
    print(vehicle_paths)
    return render_template("index.html", vehicle_image = vehicle_paths)



if __name__ == "__main__":
    app.run(debug=True)