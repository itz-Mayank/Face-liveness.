# Face-liveness.

# Face Liveness Detection using Depth Map Prediction
###![Red-_11_](https://github.com/user-attachments/assets/6af8aacc-66aa-4e36-9b1a-b94806d60b1f)
## About the Project

This is an application of a combination of Convolutional Neural Netwo
rks and Computer Vision to detect
between actual faces and fake faces in realtime environment. The image frame captured from webcam is passed over a pre-trained model. This model is trained on the depth map of images in the dataset. The depth map generation have been developed from a different CNN model.

# Create a virtual Environment
python -m venv myenv
myenv\Scripts\activate

## Requirements

* Python3
* Tensorflow
* dlib
* Keras
* numpy
* sklearn
* Imutils
* OpenCV 


## Steps to perform the authentication

![Screenshot 2024-11-09 084420](https://github.com/user-attachments/assets/6a02020c-f82a-4ab1-a248-d4697b86b20e)

Step 1: Start the Authentication Process
Start Button: Click on the Start button to initiate the face authentication process.

Step 2: Display Rules and Guidelines
Pop-up Display: A pop-up window will appear with the following instructions:

Align Your Face: Ensure your face is between 15-20 cm from the camera.

Lighting: Make sure you are in a well-lit environment to help the camera capture your face clearly.

Background: A plain background is preferred to avoid any distractions or obstructions.

Step 3: Face Authentication Begins
Follow the Red Dot: A red dot will appear on the screen. Follow the red dot with your eyes as it moves around the screen.

Read the Words: Below the red dot, a word will appear. Read out the word aloud to assist with voice recognition.

Stay Still: Try to keep your head still while moving your eyes to follow the red dot.

Step 4: Verification and Feedback
Processing: Once you complete following the red dot and reading out the word, the system will process the data.

Feedback: You will receive immediate feedback:

Success: If authentication is successful, you will be granted access.

Retry: If authentication fails, you may need to retry or follow additional instructions.

Tips for Effective Authentication
Maintain Distance: Keep your face consistently 15-20 cm from the camera.

Avoid Obstructions: Ensure nothing blocks the view of your face.

Speak Clearly: Read the words aloud and clearly for accurate voice recognition.

Troubleshooting
Camera Issues: Ensure your camera is working correctly. Try restarting the device or checking camera permissions if issues persist.

Lighting Problems: Adjust the lighting to ensure your face is well-lit. Avoid strong backlighting.

Background Noise: Ensure a quiet environment for accurate voice recognition

## The Convolutional Neural Network

The network consists of **3** hidden conlvolutional layers with **relu** as the activation function. Finally it has **1** fully connected layer.

The network is trained with **10** epochs size with batch size **32**

The ratio of training to testing bifuracation is **75:25**


