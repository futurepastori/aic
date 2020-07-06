import os
import urllib.request
from app import app
from lib.generate import generate
from flask import Flask, request, redirect, jsonify
from werkzeug.utils import secure_filename

ALLOWED_EXT = set(['jpg', 'jpeg', 'png', 'gif', 'tiff', 'webp'])

def allowed_file(filename):
	return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXT

@app.route('/upload', methods=['POST'])
def upload_image():
	if 'file' not in request.files:
		resp = jsonify({'message': 'File not uploaded'})
		resp.status_code = 400
		return resp
	file = request.files['file']
	if file.filename == '':
		resp = jsonify({'message': 'No file selected'})
		resp.status_code = 400
		return resp
	if file and allowed_file(file.filename):
		gen_message = generate()
		resp = jsonify({'message': gen_message})
		resp.status_code = 201
		return resp
	else:
		resp = jsonify({'message': 'Allowed file types are jpg, jpeg, png, gif, tiff, webp'})
		resp.status_code = 400
		return resp

@app.route('/', methods=['GET'])
def home():
	resp = jsonify({'message': 'Nothing to see around here, bud.'})
	resp.status_code = 404
	return resp

if __name__ == "__main__":
	app.run()
	
