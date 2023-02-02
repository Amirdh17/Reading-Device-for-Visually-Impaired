# Reading-Device-for-Visually-Impaired
This device looks like reading stand which can take photo of the book and read out for visually impaired person. Raspberry Pi 4 is used as controller. Major Python modules used are Pytesseract, OpenCV and Pytts.

# Description
Reading is a daily necessity to almost all human beings in contemporary times, and whoever can see, can read. But it is not the same case for visually impaired people. A system which could help in aiding the visually impaired has been developed using raspberry pi 4B, and made compact so that it could be accessible anywhere. Considering the different lighting conditions and environmental disturbances, a few image preprocessing techniques have been integrated into the design to optimize the image output. The text that is extracted from the captured image is then converted to an audio output after corrections in the text, if any, so that the users can be able to listen to what the text appears to be. A push button and ultrasonic sensor are utilized to work as the actuators of the process; to initiate the process.

# Hardware Components
*Raspberry Pi 4 Model B
*Logitech C270 HD webcam
*Ultrasonic sensor HC-SR04
*Push Button
*Adjustable Mobile Stand
*Any Audio Device

# Software Components
*Raspbian OS
*Thonny IDE
*Python Libraries (OpenCV, NumPy, Pillow, SKimage, RPi.GPIO, Pytesseract, Pyttsx3, Auto Corrector)

# Process
The process is initiated when the user presses the push button. When it is done, the ultrasonic sensor turns on and estimates the distance of the object. The object should be within the fixed threshold. The process of capturing images takes place. Once it is done the image is normalized in order to make it suitable for further processing. Even after normalization, darkness check is done to ensure that the text can be extracted without any distortion. After this step, the area of interest i.e. the region of text is identified and cropped to avoid unnecessary confusion. The image undergoes a scaling process and it is converted to a grayscale image. The process of text recognition and extraction is done using Optical Character Recognition. The resulting output is in the form of text. In order to rectify if there are any mismatched words, Natural Language Processing is made use. The text output after spell check is converted into audible output. This is interfaced with an audio device.

# Conclusion
The aim was to make a difference in the lives of the visually impaired by implementing acquired   knowledge into a practical application. Even though there are many devices present in the market to improve the lifestyle of such people, limitations such as exorbitant cost, unavailability, and lack of ease of usage prevents the access of alike devices for intended users.This device, in a major part, overcomes the above-mentioned obstacles. It was tested for its ability to act as an independent product with minimal human interference.The device  was successfully developed which superseded observed  challenges and was implemented in a more user-friendly way such that it became a standalone entity that ensured portability.
