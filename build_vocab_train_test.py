from sklearn.model_selection import train_test_split  # 数据集的划分
import os, pickle
import numpy as np

# 载入数据
x = pickle.load(open("models/IEMOCAP_all.pkl","rb"))
train_data, train_label, train_filename_save, train_text, train_label_in_num, train_label_in_onehot, train_emt = x[0], x[1], x[2], x[3], x[4], x[5],x[6]


def _word_id_map(data):
    id2word = list(set([sent for qa in data for sent in qa]))
    id2word.sort()
    id2word = ['<PAD>', '<EOS>', '<SOS>'] + id2word + ['<UNK>']
    word2id = {i[1]: i[0] for i in enumerate(id2word)}
    wordNum = len(id2word)
    print('Total words num:', len(id2word))
    return id2word,word2id,wordNum

id2word, word2id, wordNum = _word_id_map(train_text)

# 8-2比分割训练集和测试集
totalSampleNum = len(train_text)
testSize = 0.2
trainIdList, testIdList = train_test_split([i for i in range(totalSampleNum)],test_size=testSize)

chatDataWord = train_text

# 定义一个句子中的最大word数量
MaxLen = 50
def get_word_indices(data_x):
    length = len(data_x)
    return np.array([word2id[word] for word in data_x] + [0]*(MaxLen-length))[:MaxLen]

chatDataWord = np.array(chatDataWord)
# 获取训练集和测试集数据
train_text = chatDataWord[trainIdList]
test_text = chatDataWord[testIdList]
# 获取label数据
labels = np.array(train_label_in_onehot)
train_label = labels[trainIdList]
test_label = labels[testIdList]

# 构造训练集和测试集的array
train_data = {}
for i in range(len(train_text)):
    sentence_word_indices = get_word_indices(train_text[i])
    label = train_label[i]
    train_data[i] = (sentence_word_indices, label)

test_data = {}
for i in range(len(test_text)):
    sentence_word_indices = get_word_indices(test_text[i])
    label = test_label[i]
    test_data[i] = (sentence_word_indices, label)

def get_emb(local_data):
    local_text = []
    for i in range(len(local_data)):
        local_text.append(local_data[i][0])
    return np.array(local_text)

train_x = get_emb(train_data)

def get_one_hot(label):
    label_arr = [0] * wordNum
    label_arr[label]=1
    return label_arr[:]

def expendToOneHot(x):
    t = np.expand_dims(x, axis=2)
    t = t.tolist()
    for i in range(len(t)):
        for j in range(MaxLen):
            print(str(i)+'-'+str(j))
            if(t[i][j][0] == 0):
                t[i][j:] = [[0] * wordNum] * (MaxLen - j)
                break
            else:
                t[i][j] = get_one_hot(t[i][j][0])
    return np.array(t)

# 将train_x扩展1个维度，是word的数量（wordNum维的onehot）
train_x = expendToOneHot(train_x)

def get_labels(local_data):
    local_text = []
    for i in range(len(local_data)):
        local_text.append(local_data[i][1])
    return np.array(local_text)

train_y = get_labels(train_data)

test_x = get_emb(test_data)
test_y = get_labels(test_data)

test_x = expendToOneHot(test_x)

f = open('models/data_word_id_map.pkl', 'wb')
pickle.dump((train_x, train_y, test_x, test_y, id2word ,word2id), f)
f.close()

f = open('models/word_id_map.pkl', 'wb')
pickle.dump((id2word ,word2id), f)
f.close()
pass
