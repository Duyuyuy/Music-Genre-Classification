# Music-Genre-Classification
This is my thesis project

## 📔TABLE OF CONTENTS 
<details>
  <summary>Table of Contents</summary>
  <ol>
    <li><a href="#introduction">Project Overview</a></li>
    <li><a href="#process">Installation</a></li>
    <li><a href="#Usage">Usage</a></li>  
    <li><a href="#Demo">Demo</a></li>  
    <li><a href="#acknowledgements">Acknowledgments</a></li> -->
  </ol>
</details>

## Project Overview

## Installation
  Install the required dependencies by running the following command:

   ```bash
   pip install -r requirements.txt
```

## Usage
'''bash 
MyProject/

├── src/ :Contains the main source code files.

│   ├── main.py : for training model and run demo

    ├── extract.py : transform audio file to json file containing spectrogram
    
    └── prepare.py :load json array to train model
    
├── data/

    ├── Final/ : store audio file
    
         ├── ... (genres)
         
    └── JsonData/ : a dict contain: mapping, label, spectrogram
    
├── demo/

     ├── audio/
     
     └── json/
     
├── README.md

└── requirements.txt
'''
!To extract data (transform audio file to spectrogram):
```bash 
python src/extract.py
```
_extract.py
!To run the result:
```bash 
python src/main.py
```
## Configuration
_ Modify number of classes and epoch according to number of genres
_ Replace file in demo/audio/ to detect the genre of the sample

## Demo

<p align="center">
  <img src="https://github.com/Duyuyuy/Music-Genre-Classification/assets/89919775/017c7dcd-8c42-4940-9bce-9825c8c44f1f" width=800 ><br/>
</p>

