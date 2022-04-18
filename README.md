# Violence Detector

# [Project Explaination Video](https://youtu.be/xm9P7dIT4hA)

Public places are unpredictable, accidentally or with a motive violence takes place, so it is necessary to have a system that detects it automatically in real-time. To address this issue, we propose a novel method to detect violent sequences. We propose a real-time violence detection platform based on deep-learning methods. The proposed model consists of CNN and LSTM. The integral part lies in having  accuracy and fast response time.

The rise in number of violence in public assures insecurity and safety factor is endangered. Using Real-time violence detection technology which can be easily integrated with any security system can help . Its primary function is to ensure public safety through visual crowd surveillance, so any violent activity generates automatic security and alert. The violence detection module can be incorporated into the surveillance systems installed in airports, schools, parking lots, prisons, shopping malls, and other indoor and outdoor public access areas. By using machine learning trained and pretrained models it would be easy to get data about which type of violence occurs more frequently and measures can be taken to resolve it faster or before hand also area wise summary can be done to know which areas are prone towards violence to provide help at earliest

## Steps to use

### Clone git repo

```
git clone https://github.com/Shivang0/Sem-Project.git
```

### Install dependencies

```
pip install -r requirements.txt
```

### Create model or load model

To create model

```
python modelmaker.py
```

To load existing model

```
python gettingready.py
```

### Run the server

```
python main.py
```

### For testing IP Camera support
 Can directly add Real IP Camera but if not available then use Android phone or any other IP Camera
 
Install IP WebCam App for Android

[Playstore ](https://play.google.com/store/apps/details?id=com.pas.webcam&hl=en_US&gl=US)

Reduce the resolution of the camera to 320x240 for better bandwidth and performance.


