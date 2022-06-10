import os
import sys
import numpy as np
import pickle
import re
from keras.models import load_model

# Flask
from flask import Flask, redirect, url_for, request, render_template, Response, jsonify, redirect
from gevent.pywsgi import WSGIServer

# Declare a flask app
app = Flask(__name__)

# 载入word2id和id2word
data_all = pickle.load(open("models/word_id_map.pkl", "rb"))
id2word, word2id = data_all[0], data_all[1]


# 规范化数据，转小写，去除标点等其他非字母符号
def filter_data(s):
    re_han = re.compile("([a-zA-Z]+)")
    s = s.lower()
    blocks = re_han.split(s)
    txt = ''
    for blk in blocks:
        if re_han.match(blk):
            txt = txt + blk + ' '
    return txt

def get_word_indices(data_x, MaxLen):
    length = len(data_x)
    word_list = []
    for word in data_x:
        if not word in word2id:
            # 遇到词汇表之外的词就把它归为unknown
            word_list.append(word2id['<UNK>'])
        else:
            word_list.append(word2id[word])
    return np.array(word_list + [0] * (MaxLen - length))[:MaxLen]


def get_one_hot(label, wordNum):
    label_arr = [0] * wordNum
    label_arr[label] = 1
    return label_arr[:]


def expendToOneHot(x, wordNum, MaxLen):
    t = np.expand_dims(x, axis=2)
    t = t.tolist()
    for i in range(len(t)):
        for j in range(MaxLen):
            if (t[i][j][0] == 0):
                t[i][j:] = [[0] * wordNum] * (MaxLen - j)
                break
            else:
                t[i][j] = get_one_hot(t[i][j][0], wordNum)
    return np.array(t)


def processUserInput(userInputString, MaxLen):
    wordNum = len(id2word)
    # 转小写
    userInputString = userInputString.lower()
    # 按空格split成一个元素为word的list
    split_list = filter_data(userInputString).split()
    # 将word按照word2id转为id
    sentence_word_indices = get_word_indices(split_list, MaxLen)
    # 扩展维度到合适的input维度（1×MaxLen×wordNum）(1, 50, 3125)
    sentence_word_indices = np.expand_dims(sentence_word_indices, axis=0)
    final_input = expendToOneHot(sentence_word_indices, wordNum, MaxLen)
    return final_input


emotion_dict = {0: 'angry', 1: 'sad', 2: 'happy', 3: 'neutral', 4: 'frustrated'}

# 情绪识别模型
emotion_recognition_model = load_model('models/emotion_recognition_model.hdf5')
# encoder
encoder_model = load_model('models/encoder_model.hdf5')
# decoder
decoder_model = load_model('models/decoder_model.hdf5')

emotion_recognition_model._make_predict_function()
encoder_model._make_predict_function()
decoder_model._make_predict_function()
print('Model loaded. Start serving...')


def get_chatbot_response(input_seq, emotion, encoder_model, decoder_model):
    # 将输入序列进行编码
    states_value = encoder_model.predict(input_seq)
    # 获取chatbot的回应情绪
    emotion = get_one_hot(emotion, 5)

    # 生成一个size=1的空序列
    target_seq = np.array([0] * 3125 + emotion)
    target_seq = np.expand_dims(target_seq, axis=0)
    target_seq = np.expand_dims(target_seq, axis=0)

    # 将这个空序列的内容设置为开始字符
    target_seq[0, 0, word2id['<SOS>']] = 1

    # 进行字符恢复
    # 简单起见，假设batch_size = 1
    decoded_sentence = ''

    while True:
        output_tokens, state_h1, state_c1, state_h2, state_c2, state_h3, state_c3 = decoder_model.predict(
            [target_seq] + states_value)

        # sample a token
        # 去掉最后5个情绪向量
        sampled_token_index = np.argmax(output_tokens[0, -1, :])
        sampled_char = id2word[sampled_token_index]

        # 退出条件：生成 <EOS> 或者 超过最大序列长度
        if sampled_char == '<EOS>' or len(decoded_sentence) > 50:
            break

        decoded_sentence += sampled_char + " "

        # 更新target_seq
        target_seq = np.array([0] * 3125 + emotion)
        target_seq = np.expand_dims(target_seq, axis=0)
        target_seq = np.expand_dims(target_seq, axis=0)
        target_seq[0, 0, sampled_token_index] = 1.

        # 更新中间状态
        states_value = [state_h1, state_c1, state_h2, state_c2, state_h3, state_c3]

    return decoded_sentence


def inference(i, sentence, ES, ES_tmp, emotion_recognition_model, encoder_model, decoder_model, emotion_dict):
    userInput = processUserInput(sentence, 50)
    # 得到用户输入句子的情绪E
    E = emotion_recognition_model.predict(userInput)[0]
    RE = E
    # 得到chatbot的回应情绪
    UE = RE * ES_tmp * 100
    UE = UE / UE.sum()
    ES_tmp = UE
    chatbot_E = emotion_dict[UE.argmax()]

    # 根据用户输入句子和计算出来的chatbot回应情绪，使用Seq2Seq模型得到chatbot的回复句子
    input_words = processUserInput(sentence, len(filter_data(sentence).split()))
    resp_emo = UE.argmax()
    chatbot_response_sentence = get_chatbot_response(input_words, resp_emo, encoder_model, decoder_model)

    # 每进行5次对话就将ES向它的原始值恢复（向聊天机器人的一般情绪恢复）(这里i从0开始，在第五次对话chatbot做出回应后进行恢复，而不是在第6次对话开始时才恢复)
    if (i + 1) % 5 == 0:
        ES_tmp = ES_tmp * ES
        ES_tmp = ES_tmp / ES_tmp.sum()

    i += 1
    return emotion_dict[E.argmax()], chatbot_E, chatbot_response_sentence, str(ES_tmp.tolist()), i


@app.route('/', methods=['GET'])
def index():
    # Main page
    return render_template('index.html')


@app.route('/chat_angry', methods=['GET'])
def chat_angry():
    return render_template('chat.html', es='[0.96, 0.01, 0.01, 0.005, 0.015]', name='Tom', character='Angry',
                           img='../static/pics/angry.png')


@app.route('/chat_happy', methods=['GET'])
def chat_happy():
    return render_template('chat.html', es='[0.025, 0.025, 0.9, 0.025, 0.025]', name='Bob', character='Happy',
                           img='../static/pics/happy.png')


@app.route('/chat_sad', methods=['GET'])
def chat_sad():
    return render_template('chat.html', es='[0.025, 0.92, 0.005, 0.025, 0.025]', name='Susan', character='Sad',
                           img='../static/pics/sad.png')


@app.route('/chat_neutral', methods=['GET'])
def chat_neutral():
    return render_template('chat.html', es='[0.005, 0.015, 0.015, 0.96, 0.005]', name='Andy', character='Neutral',
                           img='../static/pics/neutral.png')


@app.route('/chat_frustrated', methods=['GET'])
def chat_frustrated():
    return render_template('chat.html', es='[0.1, 0.1, 0.05, 0.05, 0.7]', name='Mike', character='Frustrated',
                           img='../static/pics/frustrated.png')


@app.route('/predict', methods=['GET', 'POST'])
def predict():
    if request.method == 'POST':
        # Get the image from post request
        print('=========================')
        print(request.json)
        print('=========================')

        i = int(request.json['i'])
        sentence = request.json['userInput']

        # chatbot当前情绪强度
        ES_tmp = request.json['esTmp']
        ES_tmp = ES_tmp.split('[')[1].split(']')[0].split(',')
        for j in range(len(ES_tmp)):
            ES_tmp[j] = float(ES_tmp[j])
        ES_tmp = np.array(ES_tmp)

        # chatbot性格
        ES = request.json['es']
        ES = ES.split('[')[1].split(']')[0].split(',')
        for j in range(len(ES)):
            ES[j] = float(ES[j])
        ES = np.array(ES)

        # Make prediction
        E, chatbot_E, chatbot_response_sentence, ES_tmp, i = inference(i, sentence, ES, ES_tmp,
                                                                       emotion_recognition_model,
                                                                       encoder_model, decoder_model, emotion_dict)
        print(ES_tmp)
        # Serialize the result, you can add additional fields
        # ......

        return jsonify(i=i, E=E, chatbot_E=chatbot_E, chatbot_response_sentence=chatbot_response_sentence,
                       ES_tmp=ES_tmp)

    return None


if __name__ == '__main__':
    # app.run(port=5002, threaded=False)

    # Serve the app with gevent
    http_server = WSGIServer(('0.0.0.0', 5000), app)
    http_server.serve_forever()
