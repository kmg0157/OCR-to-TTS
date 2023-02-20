# OCR-to-TTS

This project is to develop a program that provides TTS services using OCR technology.
The reason why I started the project was to improve the service for the blind.
The OCR technology used pytesseract, and the TTS used gTTS.
I used fluoroscopic transformation and histogram normalization to solve the degradation of recognition in photographs with curvature.
This program was able to recognize text messages not only from pictures taken on the Internet but also from pictures taken in person.

The development environment is as follows
  ㆍPython 3.11.0
  ㆍOpen cv-python 4.6.0.66
  ㆍPytesseract 0.3.10
  ㆍgTTS 2.3.0
  ㆍPlaysound 1.3.0
  ㆍMatplotlib 3.6.2
  ㆍPillow 9.3.0
  
Plan to proceeding
  1. Real-time processing via IP webcam
  2. Create App Version
  3. Using Raspberry Pi to Build Smaller Devices
