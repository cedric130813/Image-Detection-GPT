from flask import Flask,render_template,request
import numpy as np
import keras.utils as image
from keras.models import load_model
from keras.applications.xception import Xception, preprocess_input,decode_predictions
# from pathlib import Path
import openai
# import os
import urllib.request

app = Flask(__name__)

@app.route("/",methods=["GET","POST"])
def index():
    result = None #added
    img_path = None
    if request.method == "POST":
        url = request.form.get("urlpath")
        if url:
            try:
                img_path, headers = urllib.request.urlretrieve(str(url))
            except:
                result = None
        if img_path:
            model = load_model('MobileNetV2_model.h5')
            img = image.load_img(img_path, target_size=(224, 224))
            image_array = image.img_to_array(img)
            x_train = np.expand_dims(image_array, axis=0)
            x_train = preprocess_input(x_train)
            prediction = model.predict(x_train)
            im_class = np.argmax(prediction[0], axis=-1)
            openai.api_key = ("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")
            response = openai.ChatCompletion.create(
                model="gpt-3.5-turbo",
                messages=[{"role":"system", "content":f"Based on the suggestion request which is and looks like {im_class}, recommend three specific items for a male runner looking for light shoes and write a short summarised product comparison of these 3 items"}],
                temperature=0.5,
                max_tokens=200)
            text = response["choices"][0]["message"]["content"]
            return(render_template("index.html",result=text, img_url=url))
        else:
            return(render_template("index.html",result="waiting"))
    else:
        return(render_template("index.html",result="waiting"))

if __name__=="__main__":
    app.run()
