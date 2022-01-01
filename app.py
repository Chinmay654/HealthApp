
# for each model 3 things should be there:- 1.) EDA 2.) Model Traning file 3.) Html file

import numpy as np 
from flask import Flask, request, render_template
import joblib
from tensorflow.keras.models import load_model
from PIL import Image
diab = joblib.load("liver.pkl")

app = Flask(__name__)


@app.route("/")
def home():
	return render_template("home.html")

@app.route("/cancer")
def cancer():
	return render_template("cancer.html")

def ValuePredictor(to_predict_list, size):
	to_predict = np.array(to_predict_list).reshape(1,size)
	if(size==5):#Cancer
		model = joblib.load('cancer_RFC_model.pkl')
		result = model.predict(to_predict)
		acc = model.predict_proba(to_predict)
		return result[0], acc
	elif(size==11):#Heart
		model = joblib.load("heart_1_LR.pkl")
		result =model.predict(to_predict)
		acc = model.predict_proba(to_predict)
		return result[0], acc
	elif(size==8):
		model = joblib.load("diabetes.pkl")
		result =model.predict(to_predict)
		acc = model.predict_proba(to_predict)
		return result[0], acc
	elif(size==10):
		model = joblib.load("liver.pkl")
		result =model.predict(to_predict)
		acc = model.predict_proba(to_predict)
		return result[0], acc


@app.route('/result',methods = ["POST"])
def result():
	if request.method == 'POST':
		to_predict_list = request.form.to_dict()
		to_predict_list=list(to_predict_list.values())
		to_predict_list = list(map(float, to_predict_list))
		print(to_predict_list)
		if(len(to_predict_list)==5):#Cancer
			result, acc  = ValuePredictor(to_predict_list,5)
		elif(len(to_predict_list)==11):
			result, acc = ValuePredictor(to_predict_list,11)
		elif(len(to_predict_list)==8):
			result, acc = ValuePredictor(to_predict_list,8)
		elif(len(to_predict_list)==10):
			result, acc = ValuePredictor(to_predict_list,10)
		
	if int(result)==1:
	  prediction ='You might be Suffering.'
	else:
	  prediction='Healthy' 
	return(render_template("result.html", prediction=prediction, accuracy=acc))

@app.route("/about")
def about():
	return render_template("about.html")

@app.route("/heart")
def heart():
    return render_template("heart.html")

@app.route("/Malaria")
def Malaria():
    return render_template("malaria1.html")

@app.route("/diabetes")
def Diabetes():
	return render_template("diabetes.html")

@app.route("/liver")
def Liver():
	return render_template("liver.html")

@app.route("/Pneumonia", methods=['GET', 'POST'])
def pneumoniaPage():
    return render_template("pneumonia.html")

@app.route("/malariapredict", methods = ['POST', 'GET'])
def malariapredictPage():
    if request.method == 'POST':
        try:
            if 'image' in request.files:
                img = Image.open(request.files['image'])
                img = img.resize((36,36))
                img = np.asarray(img)
                img = img.reshape((1,36,36,3))
                img = img.astype(np.float64)
                model = load_model("malaria_1.h5")
                pred = np.argmax(model.predict(img)[0])
        except:
            message = "Please upload an Image"
            return render_template('malaria1.html', message = message)
    return render_template('malaria_predict.html', pred = pred)

@app.route("/pneumoniapredict", methods = ['POST', 'GET'])
def pneumoniapredictPage():
    if request.method == 'POST':
        try:
            if 'image' in request.files:
                img = Image.open(request.files['image']).convert('L')
                img = img.resize((36,36))
                img = np.asarray(img)
                img = img.reshape((1,36,36,1))
                img = img / 255.0
                model = load_model("pneumonia.h5")
                pred = np.argmax(model.predict(img)[0])
        except:
            message = "Please upload an Image"
            return render_template('pneumonia.html', message = message)
    return render_template('pneumonia_predict.html', pred = pred)


if __name__ == "__main__":
	app.run(debug=True)