from flask import Flask,render_template,request
import numpy as np
import keras.utils as image
from keras.applications import xception
import openai
import urllib.request

application = Flask(__name__)

@application.route("/",methods=["GET","POST"])
def index():
    img_path = None
    if request.method == "POST":
        url = request.form.get("urlpath")
        if url:
            try:
                img_path, headers = urllib.request.urlretrieve(str(url))
            except:
                pass
        if img_path:
            img = image.load_img(img_path, target_size=(299, 299))
            image_array = image.img_to_array(img)
            x_train = np.expand_dims(image_array, axis=0)
            x_train = xception.preprocess_input(x_train)
            model = xception.Xception(weights="imagenet")
            prediction = model.predict(x_train)
            pred = xception.decode_predictions(prediction)[0][0]
            openai.api_key = ("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")
            response = openai.Completion.create(
                model="text-curie-001", #text-curie-001,text-davinci-003 https://beta.openai.com/docs/models/gpt-3
                prompt=f"I want you to act as an retail recommendation system. I will provide you with a list of recommended items, and you will suggest 5 similar items including its brand that a customer could buy in Singapore based on those items, of different brands and bracket with its retail price in Singapore dollars. My suggestion request is {pred[1]}",
                temperature=0.5,
                max_tokens=400)
            text = response['choices'][0]['text']
            return(render_template("index.html",result=text, img_url=url))
        else:
            return(render_template("index.html",result="waiting"))
    else:
        return(render_template("index.html",result="waiting"))

if __name__=="__main__":
    application.run()
