# Image-Classification-Streamlit-TensorFlow
A basic web-app for image classification using Streamlit and TensorFlow.

# It classifies the given image of a car into one of the following fourteen categories :-  
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

<img src ='misc/sample_home_page.jpg' width = 700>  

<img src ='misc/sample_output.jpg' width = 700>


## Notes
* A simple car classification model was trained using TensorFlow.  
* The weights are stored as `my_model.hdf5`.  
* The code to train the modify and train the model can be found in `car-images-classification-using-cnn.ipynb`.  
* The web-app created using Streamlit can be found in `app.py`


## References

* https://www.tensorflow.org/tutorials/images/classification
* https://docs.streamlit.io/en/stable/
