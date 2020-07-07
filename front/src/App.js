import React, {useState} from 'react';
import axios from 'axios';
import 'normalize.css';
import './App.scss';


function App() {
	const [isLoading, setLoadingState] = useState(false)
	const [selectedImage, setSelectedImage] = useState(null)
	const [previewImage, setPreviewImage] = useState("")

	const [response, setResponse] = useState({})

	const onSubmitHandler = (e) => {
		console.log('submithandler')
		e.preventDefault();
		const data = new FormData();
		data.append('file', selectedImage);
		submitForm(data);
	}

	const onChangeHandler = (e) => {
		setSelectedImage(e.target.files[0]);
		setPreviewImage(URL.createObjectURL(e.target.files[0]))
	}

	const submitForm = async (data) => {
		console.log('submitform')
		setLoadingState(true);
		axios.post("https://cors-anywhere.herokuapp.com/http://d4901335b230.ngrok.io/upload", data, {})
					.then(res => {
						setResponse(res.data)
						setLoadingState(false);
					})
	}

	return (
		<div className="app">
			<main>
			<header>
				<div className="title">
					<h1>Automatic Image Captioning demo</h1>
					<small>Víctor Ramírez</small>
				</div>
				<nav className="nav">
					<a href="https://www.github.com/futurepastori/aic-demo.git">Source</a>
				</nav>
			</header>
			<article>
				<p>Upload a photo from your device. It <strike>may</strike> will take a while.</p>
				<div>
					<fieldset>
						<div className="file-uploader">
							<label htmlFor="file">Image file (JPG, PNG, GIF or TIFF)</label>
							<input type="file" name="file" onChange={e => onChangeHandler(e)} />
						</div>
						{!isLoading ? <button className="submit" onClick={e => onSubmitHandler(e)}>Get captions</button> : <span>Loading...</span>}
					</fieldset>
				</div>
				<section className="preview-container">
					{selectedImage !== null && <div className="preview">
						<img className="img-preview" src={previewImage}/>
					</div>}
				</section>
				{Object.keys(response).length !== 0 && <ul className="results">
					{Object.keys(response).map((key, i) => 
						<li key={i} className="result">
							<span className="method">{key}</span>
							<span className="caption">{response[key]}</span>
						</li>
					)}
				</ul>}
			</article>
			</main>
		</div>
	);
}

export default App;
