# Breast-Cancer-Image-Segmentation-Streamlit-TensorFlow
A basic web-app for image classification using Streamlit and TensorFlow.

# It classifies the given image of Breast-Cancer-Image-Segmentation into one of the following tree categories :-  
* Normal.
* malignant.
* normal.

## Commands

To run the app locally, use the following command :-  
`streamlit run app.py`  

The webpage should open in the browser automatically.  
If it doesn't, the local URL would be output in the terminal, just copy it and open it in the browser manually.  
By default, it would be `http://localhost:8501/`  

Click on `Browse files` and choose an image from your computer to upload.  
Once uploaded, the model will perform inference and the output will be displayed.  

## Output

<img src ='misc/sample_home_page.jpeg' width = 700>  

<img src ='misc/sample_output.jpeg' width = 700>


## Notes
* A simple Breast-Cancer-Image classification model was trained using TensorFlow.  
* The weights are stored as `my_model.h5`.  
* The code to train the modify and train the model can be found in `breast-cancer-image-segmentation-cnn.ipynb`.  
* The web-app created using Streamlit can be found in `app.py`


## References

* https://www.tensorflow.org/tutorials/images/classification
* https://docs.streamlit.io/en/stable/
