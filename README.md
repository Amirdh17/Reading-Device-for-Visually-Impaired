# Reading-Device-for-Visually-Impair
This device looks like reading stand which can take photo of the book and read out for visually impaired person. Raspberry Pi 4 is used as controller. Major Python modules used are Pytesseract, OpenCV and Pytts.

#Description
Reading is a daily necessity to almost all human beings in contemporary times, and whoever can see, can read. But it is not the same case for visually impaired people. A system which could help in aiding the visually impaired has been developed using raspberry pi 4B, and made compact so that it could be accessible anywhere. Considering the different lighting conditions and environmental disturbances, a few image preprocessing techniques have been integrated into the design to optimize the image output. The text that is extracted from the captured image is then converted to an audio output after corrections in the text, if any, so that the users can be able to listen to what the text appears to be. A push button and ultrasonic sensor are utilized to work as the actuators of the process and to initiate the process.

#Hardware Components
*Raspberry Pi 4 Model B
*Logitech C270 HD webcam
*Ultrasonic sensor HC-SR04
*Push Button
*Adjustable Mobile Stand
*Any Audio device

#Software Components
*Raspbian OS
*Thonny IDE
*Python Libraries (OpenCV, NumPy, Pillow, Skimage, RPi.GPIO, Pytesseract, Pyttsx3, Auto Corrector)

#Methodology
1. Connect components to the Pi
2. Required python libraries were installed
3. Push button setup - sensor is activated 
4. Capture and process images to acquire text
5. Convert text to speech 
6. Acquire speech output 
7. Compact device formulation was done

