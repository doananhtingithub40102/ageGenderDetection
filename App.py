import numpy as np
import cv2
import os
from flask import Flask, render_template, request, redirect, url_for, Response
from PIL import Image
from werkzeug.utils import secure_filename
from mtcnn.mtcnn import MTCNN
from detect import predict, draw_box_faces, crop_faces
import tensorflow as tf


def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


def getRangeAge(pred_age):
    arrRangeAge = ["1-14", "15-25", "26-40", "41-60", "61-116"]
    maxIndex = list(pred_age[0]).index(max(pred_age[0]))
    rangeAge = arrRangeAge[maxIndex]
    
    return rangeAge


def getGender(pred_sex):
    index_maxValue = pred_sex.argmax()
    gender = "Male"

    if index_maxValue == 1:
        gender = "Female"

    return gender


def getPercentOfAge(pred_age):
	maxIndex = list(pred_age[0]).index(max(pred_age[0]))
	percent = str(round(pred_age[0][maxIndex] * 100, 2)) + "%"

	return percent


def getPercentOfGender(pred_sex):
    index_maxValue = pred_sex.argmax()
    pred_gender = pred_sex[0][0]

    if index_maxValue == 1:
        pred_gender = pred_sex[0][1]

    percent = str(round(pred_gender * 100, 2)) + "%"

    return percent


def draw_box_and_label(img, box, conf, predAge, predGender):
	(startX, startY, endX, endY) = box
	y = startY - 60 if startY - 60 > 10 else startY + 15
	age = getRangeAge(predAge)
	percent_age = getPercentOfAge(predAge)
	gender = getGender(predGender)
	percent_gender = getPercentOfGender(predGender)

	cv2.rectangle(img, (startX, startY), (endX, endY),
					(0, 255, 0), 2)
	cv2.putText(img, conf, (startX, y),
		cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 0, 0), 2)
	cv2.putText(img, age, (startX, y + 25),
		cv2.FONT_HERSHEY_SIMPLEX, 0.60, (0, 0, 255), 2)
	cv2.putText(img, percent_age, (startX + 80, y + 25),
		cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 0, 0), 2)
	cv2.putText(img, gender, (startX, y + 50),
		cv2.FONT_HERSHEY_SIMPLEX, 0.60, (0, 0, 255), 2)
	cv2.putText(img, percent_gender, (startX + 80, y + 50),
		cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 0, 0), 2)

	return img


def gen_frames():  # generate frame by frame from camera
	global camera
	if not camera.isOpened():
		camera = cv2.VideoCapture(cv2.CAP_V4L2)

	while True:
        # Capture frame-by-frame
		success, frame = camera.read()  # read the camera frame
		if not success:
			print("Can't receive frame (stream end?). Exiting ...")
			break
		else:
			# grab the frame dimensions and convert it to a blob
			(h, w) = frame.shape[:2]
			blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)), 1.0,
				(300, 300), (104.0, 177.0, 123.0))

			# pass the blob through the network and obtain the detections and
			# predictions
			net.setInput(blob)
			detections = net.forward()

			# loop over the detections
			for i in range(0, detections.shape[2]):
				# extract the confidence (i.e., probability) associated with the
				# prediction
				confidence = detections[0, 0, i, 2]
				# filter out weak detections by ensuring the `confidence` is
				# greater than the minimum confidence
				if confidence < 0.5:
					continue
				# compute the (x, y)-coordinates of the bounding box for the
				# object
				box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
				(startX, startY, endX, endY) = box.astype("int")

				face = frame[startY:endY, startX:endX, :]
				face = cv2.resize(face, (48, 48), cv2.INTER_AREA)
				face = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)

				pixels = np.zeros((48, 48, 3))
				pixels[:, :, 0] = pixels[:, :, 1] = pixels[:, :, 2] = face
				pred = multitask_model.predict(np.array([pixels]))
		
				# draw the bounding box of the face along with the associated
				# probability
				conf = "{:.2f}%".format(confidence * 100)
				frame = draw_box_and_label(frame, (startX, startY, endX, endY), conf, pred[0], pred[1])
			
			ret, buffer = cv2.imencode('.jpg', frame)
			frame = buffer.tobytes()
			yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')  # concat frame one by one and show result


app = Flask(__name__)

UPLOAD_FOLDER = 'static/uploads/'

app.secret_key = "secret key"
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024

ALLOWED_EXTENSIONS = set(['png', 'jpg', 'jpeg'])

INDEX_FACE = 0
filename = ""
faceFiles = []
output = []

# load model detect faces
detect_model = MTCNN()
# load our serialized model from disk
net = cv2.dnn.readNetFromCaffe("./saved_model/deploy.prototxt.txt", "./saved_model/res10_300x300_ssd_iter_140000_fp16.caffemodel")
# load model multitask learning
multitask_model = tf.keras.models.load_model('./saved_model/efficientNetB2_weight.h5')

streaming = True
camera = cv2.VideoCapture(cv2.CAP_V4L2)

@app.route('/')
def index_view():
	return render_template('index.html', imgFile="detectedFace.jpg", faceFile="face.jpg", gender="Male", age="26-40 years old")

@app.route('/', methods = ['POST'])
def result():
	global INDEX_FACE
	global filename
	global faceFiles
	global output

	INDEX_FACE = 0
	faceFiles = []
	output = []
	keep = ("detectedFace.jpg", "face.jpg")

	for root, dirs, files in os.walk(app.config['UPLOAD_FOLDER']):
		for f in files:
			if f in keep:
				continue
			else:
				path = os.path.join(root, f)
				os.remove(path)

	if 'file' not in request.files:
		return redirect(request.url)
	file = request.files['file']
	if file.filename == '':
		return redirect(request.url)
	if file and allowed_file(file.filename):
		filename = secure_filename(file.filename)
		path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
		file.save(path)
		
		loadImg = Image.open(path)
		pixels = cv2.imread(path)
		pixels = cv2.cvtColor(pixels, cv2.COLOR_RGBA2BGR)

		faces = detect_model.detect_faces(pixels)
		if faces == []:
			loadImg.close()
			os.remove(path)
			return redirect(request.url)

		cropped_images = crop_faces(loadImg, faces)
		arrFile = filename.split(".")

		for i, face in enumerate(cropped_images):
			faceFile = arrFile[0] + "_face" + str(i) + "." + arrFile[1]
			faceFiles.append(faceFile)
			path = os.path.join(app.config['UPLOAD_FOLDER'], faceFile)
			face.save(path)

			pixels = cv2.imread(path)
			pixels = cv2.cvtColor(pixels, cv2.COLOR_BGR2GRAY)
			pixels = cv2.resize(pixels, (48, 48))
			
			# Predict age, gender
			age, gender = predict(pixels)
			output.append((age, gender))

		draw_box_faces(loadImg, faces)
		
		filename = arrFile[0] + "_detectedFaces" + "." + arrFile[1]
		path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
		loadImg.save(path)

		return render_template('index.html', imgFile=filename, faceFile=faceFiles[INDEX_FACE], age=output[INDEX_FACE][0], gender=output[INDEX_FACE][1])

@app.route('/display/<filename>')
def display_image(filename):
    return redirect(url_for('static', filename='uploads/' + filename), code=301)

@app.route('/webcam', methods = ['POST'])
def webcam_view():
	return render_template('webcam.html')

@app.route('/video_feed')
def video_feed():
    #Video streaming route. Put this in the src attribute of an img tag
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/stop')
def stop():
	if camera.isOpened():
		print("Releasing cam feed")
		camera.release()
	return ""

@app.route('/next_face')
def next_face():
	global INDEX_FACE
	if len(faceFiles) == 0:
		return render_template('index.html', imgFile="detectedFace.jpg", faceFile="face.jpg", gender="Male", age="26-40 years old")
	
	if INDEX_FACE <= (len(faceFiles) - 2):
		INDEX_FACE += 1
	else:
		INDEX_FACE = 0

	return render_template('index.html', imgFile=filename, faceFile=faceFiles[INDEX_FACE], age=output[INDEX_FACE][0], gender=output[INDEX_FACE][1])

if __name__ == '__main__':
    app.run(debug=True, port=8000, host='0.0.0.0')
