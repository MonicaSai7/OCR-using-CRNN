# OCR-using-CRNN
Optical Character Recognition (OCR) is a computer vision technique which recognizes text present in any form of
images, such as scanned documents and photos. OCR is used to transform the printed text present in the scanned
images into digital form. In the recent years, OCR has improved significantly in accurate recognition of text from
images. 

The OCR system in this project is being developed using CRNN deep learning architecture which is a combination of CNN and RNN.
Initially MJSynth Synthetic word dataset is used to develop the model.

## Environment Setup
Run the following command in the app directory: <br>
`pip install -r requirements.txt`

## Run
Run the flask application using: <br>
`python app.py`

## Run Docker image
Access the Docker image from the provided Google Drive link adn run the following command:<br>
https://drive.google.com/file/d/14Ksd35OnBjnelJToi6bjrdLz7AYncRcz/view?usp=sharing <br>
`sudo docker run -p 8081:8001 ocr-docker`
