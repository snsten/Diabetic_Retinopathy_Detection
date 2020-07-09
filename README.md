# Diabetic Retinopathy Web App
Webapp for classification of Diabetic Retinopathy from retinal images using flask and keras.
<p align="center">
  <img src="https://github.com/snsten/Diabetic-Retinopathy-WebApp/blob/master/data/prediction.jpg">
</p>

## Diabetic Retinopathy
Diabetic retinopathy is an eye condition that can cause vision loss and blindness in people who have diabetes. It affects blood vessels in the retina.

No Diabetic Retinopathy    |  Severe Diabetic Retinopathy
:-------------------------:|:-------------------------:
![](https://github.com/snsten/Diabetic-Retinopathy-WebApp/blob/master/data/no_dr.jpg)  |  ![](https://github.com/snsten/Diabetic-Retinopathy-WebApp/blob/master/data/severe_dr.jpg)

## Data Description

Dataset consists of retina images taken using fundus photography under a variety of imaging conditions.

A clinician has rated each image for the severity of diabetic retinopathy on a scale of 0 to 4:

    0 - No DR

    1 - Mild

    2 - Moderate

    3 - Severe

    4 - Proliferative DR

## About the Model used for prediction
### Densely Connected Convolutional Networks or DenseNet 

Dense Convolutional Network (DenseNet), which connects each layer to every other layer in a feed-forward fashion. Whereas traditional convolutional networks with L layers have L connections - one between each layer and its subsequent layer.

A more efficient model variant DenseNet-BC (DenseNet-Bottleneck-Compressed) networks are trained. Using the DenseNet-BC-121-32 model as the base model.

Advantages of DenseNet-BC are:
 - Reduced number of parameters
 - Similar or Better performance
 - Better accuracy
 
 
 Dense Net architecture as shown in the original paper which shows the connections from each layer to every other layer:
 
 <p align="center">
  <img src="https://github.com/snsten/Diabetic-Retinopathy-WebApp/blob/master/data/densenet.jpg">
</p>

## Requirements Insatallation
### Using conda or virtualenv
```
virtualenv venv
source /bin/activate
python3 -m pip install -r requirements.txt
```
### Direct (Not recommended)
`python3 -m pip install -r requirements.txt`

## Run the Webapp by executing 
`python3 app.py`

## Refrences
- [Diabetic Retinopathy Wikipedia](https://en.wikipedia.org/wiki/Diabetic_retinopathy)
- [Diabetic Retinopathy National Eye Institute](https://www.nei.nih.gov/learn-about-eye-health/eye-conditions-and-diseases/diabetic-retinopathy)
- [Fundus Image Dataset from Kaggle](https://www.kaggle.com/c/aptos2019-blindness-detection/data)
- [Densely Connected Convolutional Networks (DenseNets)](https://github.com/liuzhuang13/DenseNet)
