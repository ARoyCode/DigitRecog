# DigitRecog

The web app allows user to submit a handwritten digit in the form of a picture and output the recognition result of [0..9].
The web Micro Framework is developed in Flask.
We TensorFlow using the Keras API to train a CNN with MNIST dataset. Data augmentaion has been done using Keras ImageDataGenerator.

Dependencies: Install the dependencies using command: pip install requirements.txt

Usage: Once dependencies are installed, run the command: python main_app.py

To run on localhost change main_app.py as:  app.run(port=5000, debug=True)

To listen on all public IPs:  app.run(host='0.0.0.0')

To listen on a particular IP:  app.run(host='192.168.1.9')
