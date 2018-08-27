import json
from flask import request
from flask import jsonify
import splitwords
from flask import Flask

app = Flask(__name__)
@app.route('/segment',methods=['POST',])
def index():
    data = json.loads(request.data)
    words = data['words']
    result=[]
    for word in words:
        if word.strip()!='':
            res= splitwords.model.interactive_shell(splitwords.processing_word, word,splitwords.sess)
            result.append(res)
        else:
            res=[]
            res.append('')
            result.append(res)

    return jsonify({"result":result})

if __name__ == '__main__':
    app.run(host='0.0.0.0',port=1234,debug=False)