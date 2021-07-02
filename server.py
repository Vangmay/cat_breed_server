from flask import Flask, request, jsonify
import impp 

app = Flask(__name__)

@app.route('/classify_cat',methods=['get','post'])
def classify_cat():
    image_data = request.form['image_data']

    response = jsonify(impp.send_predictions(image_data))

    response.headers.add('Access-Control-Allow-Origin','*')
    return response,'hello world'

if __name__ == '__main__':
    print('wow')
    impp.load_stuff()
    app.run()
