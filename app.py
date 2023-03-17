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
            response = openai.Completion.create(
                model="text-curie-001", #text-curie-001,text-davinci-003 https://beta.openai.com/docs/models/gpt-3
                prompt=f"I want you to act as an retail recommendation system. Based on the suggestion request, you will suggest three very similar retail items including its brand that a customer in Singapore could buy together with the suggestion request. My suggestion request looks like {im_class}",
                temperature=0.7,
                max_tokens=400)
            text = response['choices'][0]['text']
            return(render_template("index.html",result=text, img_url=url))
        else:
            return(render_template("index.html",result="waiting"))
    else:
        return(render_template("index.html",result="waiting"))

if __name__=="__main__":
    app.run()
