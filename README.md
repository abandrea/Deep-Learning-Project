# Deep-Learning-Project
Deep Learning Project - Skin Lesion Classification with Explainability

## Project Overview
This project aims to classify 7 different types of skin lesions using deep learning models. The dataset used in this project is the HAM10000 dataset, with metadata and images of skin lesions.

## General Idea
Clinicians classify skin lesions using multiple methods. The process begins with visual inspection and dermatoscopy, where they examine characteristics like size, shape, color, and subsurface features using specialized magnification tools. For suspicious lesions, histopathological analysis of tissue samples provides detailed cellular information. Modern classification increasingly incorporates advanced imaging systems and AI algorithms, which analyze lesion patterns against databases to support diagnosis. This combined approach enables accurate lesion classification and appropriate treatment planning.

What if we could build a deep learning model that can classify skin lesions with high accuracy and provide explanations for its predictions? This project aims to do just that. By training a deep learning model on the HAM10000 dataset, we can classify skin lesions into 7 different classes and provide explanations for the model's predictions using techniques like Grad-CAM. This project combines the power of deep learning with explainability to create a robust skin lesion classification system.

## About the Project
The project is divided into 3 main parts:
1. EDA and Data Preprocessing
2. Model Building
3. Model Explainability

## EDA and Data Preprocessing
The dataset is first preprocessed to extract the necessary information from the metadata. After combining the metadata with the images, the final dataset used contains 5514 images with 7 different classes of skin lesions, each with information about the patient (age, sex, lesion type, lesion location). 

![Classes](images/class.png)

![Pie Chart](images/pie.png)

### Types of Skin Lesions
The dataset contains 7 different types of skin lesions:
1. Melanocytic nevi (`nv`): Commonly known as moles, these benign growths are usually well-defined, round or oval, and can be flat or raised. Their color ranges from flesh-toned to dark brown.
2. Melanoma (`mel`): A malignant tumor originating from melanocytes. Key warning signs include asymmetry, irregular borders, multiple colors, diameter larger than 6mm, and any evolution in size, shape, or color.
3. Benign keratosis-like lesions (`bkl`): This category encompasses seborrheic keratoses, solar lentigines, and lichen-planus-like keratoses. Seborrheic keratoses often have a "stuck-on" appearance with a waxy or wart-like surface. Solar lentigines, commonly known as age spots, are flat, brown patches resulting from sun exposure.
4. Basal cell carcinoma (`bcc`): A common form of skin cancer that rarely metastasizes but can grow destructively if untreated. It appears in different morphologic variants, including flat, nodular, pigmented, and cystic forms.
5. Actinic keratoses (`akiec`): These are rough, scaly patches that develop from prolonged sun exposure and can progress to squamous cell carcinoma. They often feel like sandpaper and may be easier to feel than see.
6. Vascular lesions (`vasc`): This group includes cherry angiomas, angiokeratomas, and pyogenic granulomas. They are typically red or purple due to blood vessel proliferation and can vary in size and shape.
7. Dermatofibroma (`df`): A benign skin lesion that is firm, often brownish, and may show a central zone of fibrosis. They are usually asymptomatic but can be tender when pressed.






<!--
### Sample Images from Each Class
Here are some sample images from each of the 7 classes of skin lesions:

#### Melanocytic nevi (nv)
![Melanocytic nevi](images/nv_sample.jpg)

#### Melanoma (mel)
![Melanoma](images/mel_sample.jpg)

#### Benign keratosis-like lesions (bkl)
![Benign keratosis-like lesions](images/bkl_sample.jpg)

#### Basal cell carcinoma (bcc)
![Basal cell carcinoma](images/bcc_sample.jpg)

#### Actinic keratoses (akiec)
![Actinic keratoses](images/akiec_sample.jpg)

#### Vascular lesions (vasc)
![Vascular lesions](images/vasc_sample.jpg)

#### Dermatofibroma (df)
![Dermatofibroma](images/df_sample.jpg)

-->