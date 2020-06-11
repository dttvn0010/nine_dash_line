import cv2
import numpy as np
from flask import Flask, request, render_template
from keras.models import Model, model_from_json
import tensorflow as tf
graph = tf.get_default_graph()

input_size = 256

with open('model/model.json') as f: 
    model = model_from_json(f.read())        
       
model.load_weights('model/model.h5')

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')
    
@app.route('/upload', methods=['POST'])
def upload():
    file = request.files.get('img')
    result = 'OK'
    
    if file and file.name:
        file.save('tmp.jpg')
        img = cv2.imread('tmp.jpg')
                
        if img is None:
            result = "Unsuportted format file"
        else:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            h, w, _ = img.shape
            
            if w/h > 1.3:
                img1 = cv2.resize(img[:,:h], (input_size, input_size))
                img2 = cv2.resize(img[:,-h:], (input_size, input_size))
                inputs = np.array([img1, img2])        
            else:            
                img = cv2.resize(img, (input_size, input_size))
                inputs = np.array([img])
            
            with graph.as_default():
                outputs = model.predict(inputs) 
                error = np.max(outputs, axis=0)[1]
                
                if error > 0.55:
                    result = "Error"
                elif error > 0.2:
                    result = "May be error"
        
    return f"<a href='/'>Back</a>&nbsp; {result}"
    
app.run(debug=True)    