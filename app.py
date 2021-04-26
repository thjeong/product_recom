# import the necessary packages
import os, logging
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from flask import Flask, request, jsonify, redirect

# initialize our Flask application and the Keras model
app = Flask(__name__)
RANK_TO_SERVE = 5
N_RANDOM_IF_NO_USER_SELECTION = 3
nn = None
ftrs, ftr_array, ftr_ids = None, None, None

class NeuralNetwork:
    def __init__(self):
        self.session = tf.Session()
        self.graph = tf.get_default_graph()
        # the folder in which the model and weights are stored
        self.model_folder = os.path.join(os.path.abspath("src"), "static")
        self.model = None
        # for some reason in a flask app the graph/session needs to be used in the init else it hangs on other threads
        with self.graph.as_default():
            with self.session.as_default():
                logging.info("neural network initialised")
    
    def load(self, file_name=None):
        """
        :param file_name: [model_file_name, weights_file_name]
        :return:
        """
        with self.graph.as_default():
            with self.session.as_default():
                try:
                    self.model = keras.models.load_model(file_name)
                    return True
                except Exception as e:
                    logging.exception(e)
                    return False

    def predict(self, x):
        with self.graph.as_default():
            with self.session.as_default():
                y = self.model.predict(x)
        return y

def load_ftrmap():
    # load feather mapper to map CARD_ID to FEATURE ARRAY
    global ftrs, ftr_array, ftr_ids
    card_df = pd.read_csv('./bin/20210421_card_meta_features_normalized.csv') #, index=False)
    card_df = card_df.set_index('상품번호')

    # 모델 입력용 array
    ftr_array = card_df.to_numpy()
    ftrs = {x:ftr_array[idx] for idx, x in enumerate(card_df.index)}
    ftr_ids = np.array(list(ftrs.keys()))


from numpy.linalg import norm
def cos_sim(X=[]):
    A, B = X
    return np.matmul(A, B.T)/(norm(A)*norm(B, axis=1))
    #args = np.argsort(mat)[-rank:]
    #return mat[args], ftr_ids[args]

def make_prediction_merge_3_results(data):
    """Data Format:
    data = [
    {'cards': ["BOACF4", "BOAC6A", "BNBC47", "AXAAZE", "BPAC88"], 'actions': [0, 0, 0, 1, 0]},
    {'cards': ["BECB97", "BOAC6A", "BNBC47", "AXAAZE", "BPAC88"], 'actions': [1, 0, 0, 0, 0]},
    {'cards': ["AXDBMR", "BOAC6A", "BCBBLO", "AZAAZW", "AXAAZE"], 'actions': [0, 0, 0, 1, 0]}
    ]"""
    ret = []
    for actions in data:
        #tmp = actions['cards'] * 
        input = [ftrs.get(actions['cards'][idx[0]]) for idx in np.argwhere(actions['actions'])]
        input = input if len(input) > 0 else ftr_array[np.random.choice(len(ftr_array), N_RANDOM_IF_NO_USER_SELECTION)]
        X1, X2 = np.mean(input, axis=0), np.max(input, axis=0)
        #print(X1, X2)
        #pred = model.predict([pd.DataFrame(X1).T, pd.DataFrame(X2).T])
        pred = nn.predict([pd.DataFrame(X1).T, pd.DataFrame(X2).T])
        #print(pred)
        A = np.concatenate(pred, axis=1)[0]
        mat = cos_sim([A, ftr_array])
        args = np.argsort(mat)[-RANK_TO_SERVE:]
        ret.append([mat[args], ftr_ids[args]])
    _score, _ids = np.concatenate(ret, axis=1)
    # print(_score, _ids)
    recom = []
    for idx in reversed(np.argsort(_score)):
        if _ids[idx] in recom: continue
        recom.append(_ids[idx])
        if len(recom) == RANK_TO_SERVE: break
    return recom

@app.route("/recom", methods=["POST"])
def predict():
    # initialize the data dictionary that will be returned from the
    # view
    print('[requested json]', request)
    if not request.is_json:
        return jsonify({"msg": "Missing JSON in request"}), 400

    req = request.get_json()
    # User_Actions contains 3 different trials user did.
    return jsonify(make_prediction_merge_3_results(req['User_Actions']))
        

# if this is the main thread of execution first load the model and
# then start the server
if __name__ == "__main__":
    print(("* Loading Keras model and Flask starting server..."
        "please wait until server has fully started"))
    #global nn
    nn = NeuralNetwork()
    nn.load('./bin/20210426.model')
    load_ftrmap()
    app.run()