import os
import glob
import python_speech_features as ps
import wave
import numpy as np
import re
import jieba
import keras as kr
import pickle


def read_file(filename):
    file = wave.open(filename, 'r')
    params = file.getparams()
    nchannels, sampwidth, framerate, wav_length = params[:4]
    str_data = file.readframes(wav_length)
    wavedata = np.fromstring(str_data, dtype=np.short)
    # wavedata = np.float(wavedata*1.0/max(abs(wavedata)))  # normalization)
    time = np.arange(0, wav_length) * (1.0 / framerate)
    file.close()
    return wavedata, time, framerate


def generate_label(emotion):
    # 5种不同情绪
    if (emotion == 'ang'):
        label = 0
    elif (emotion == 'sad'):
        label = 1
    elif (emotion == 'hap'):
        label = 2
    elif (emotion == 'neu'):
        label = 3
    elif (emotion == 'fru'):
        label = 4
    else:
        label = 5
    return label


def read_IEMOCAP():
    # 存放IEMOCAP数据集的地址(The address where the IEMOCAP dataset is stored)
    rootdir = r"E:\Chrome_dl\IEMOCAP\IEMOCAP_full_release"
    train_emt = {'hap': 0, 'ang': 0, 'neu': 0, 'sad': 0, 'fru': 0}
    train_filename_save = []
    train_data = []
    train_label = []

    for speaker in os.listdir(rootdir):
        # eg. rootdir：E:\Chrome_dl\IEMOCAP\IEMOCAP_full_release
        # 遍历每个session
        if (speaker[0] == 'S'):
            sub_dir = os.path.join(rootdir, speaker, 'sentences\\wav')
            print(sub_dir)
            print('======================')
            emoevl = os.path.join(rootdir, speaker, 'dialog\\EmoEvaluation')
            # sess是对话的文件，这个for循环中循环的对象是对话，sub_dir里面是一个session中的全部对话文件
            # eg. sub_dir：E:\\Chrome_dl\\IEMOCAP\\IEMOCAP_full_release\\Session1\\sentences\\wav
            # eg. sess：Ses01F_impro01（这是一个对话的文件夹）
            for sess in os.listdir(sub_dir):
                print(sess)
                print('=================')
                emotdir = emoevl + '/' + sess + '.txt'
                emot_map = {}
                with open(emotdir, 'r') as emot_to_read:
                    while True:
                        line = emot_to_read.readline()
                        if not line:
                            break
                        if (line[0] == '['):
                            t = line.split()
                            emot_map[t[3]] = t[4]

                file_dir = os.path.join(sub_dir, sess, '*.wav')
                # files是一整个对话，里面包括这段对话中的所有utterance
                '''
                eg. files：['E:\\Chrome_dl\\IEMOCAP\\IEMOCAP_full_release\\Session1\\sentences\\wav\\Ses01F_impro01\\Ses01F_impro01_F000.wav',
 'E:\\Chrome_dl\\IEMOCAP\\IEMOCAP_full_release\\Session1\\sentences\\wav\\Ses01F_impro01\\Ses01F_impro01_F001.wav',
 ...]
                '''
                files = glob.glob(file_dir)
                
                for filename in files:
                    # 对一个对话中的每一句utterance进行操作（记录对话内容和对应的情绪）
                    # eg. filename：E:\\Chrome_dl\\IEMOCAP\\IEMOCAP_full_release\\Session1\\sentences\\wav\\Ses01F_impro01\\Ses01F_impro01_F000.wav
                    wavname = filename.split("\\")[-1][:-4]
                    emotion = emot_map[wavname]

                    if emotion in ['hap', 'ang', 'neu', 'sad', 'fru']:
                        data, time, rate = read_file(filename)
                        mel_spec = ps.logfbank(data, rate, nfilt=40)
                        time = mel_spec.shape[0]
                        if time <= 300:
                            part = mel_spec
                            part = np.pad(part, ((0, 300 - part.shape[0]), (0, 0)), 'constant',
                                          constant_values=0)
                            train_data.append(part)
                            train_label.append(emotion)
                            train_emt[emotion] = train_emt[emotion] + 1
                            train_filename_save.append(filename)
                        else:
                            begin = 0
                            end = 300
                            part = mel_spec[begin:end, :]
                            train_data.append(part)
                            train_label.append(emotion)
                            train_emt[emotion] = train_emt[emotion] + 1
                            train_filename_save.append(filename)
                    elif emotion in ['exc']:  # 将激动也归为happy类
                        data, time, rate = read_file(filename)
                        mel_spec = ps.logfbank(data, rate, nfilt=40)
                        time = mel_spec.shape[0]
                        if time <= 300:
                            part = mel_spec
                            part = np.pad(part, ((0, 300 - part.shape[0]), (0, 0)), 'constant',
                                          constant_values=0)
                            train_data.append(part)
                            train_label.append('hap')
                            train_emt['hap'] = train_emt['hap'] + 1
                            train_filename_save.append(filename)
                        else:
                            begin = 0
                            end = 300
                            part = mel_spec[begin:end, :]
                            train_data.append(part)
                            train_label.append('hap')
                            train_emt['hap'] = train_emt['hap'] + 1
                            train_filename_save.append(filename)

    # 从transcription文件夹中提取出所有文本句子
    train_text = []
    i = 0
    # 这里的每一个path都对应着一个对话文件
    for path in train_filename_save:
        temp_1 = path.split("\\")
        # eg. sub_path：E:\\Chrome_dl\\IEMOCAP\\IEMOCAP_full_release\\Session1\\dialog\\transcriptions\\Ses01F_impro01.txt
        sub_path = temp_1[0] + "\\" + temp_1[1] + "\\" + temp_1[2] + "\\" + temp_1[3] \
                   + "\\" + temp_1[4] + "\\" + 'dialog\\transcriptions\\' + temp_1[7] + ".txt"
        with open(sub_path, 'r') as f:
            d = f.read()
            txt = d.split("\n")
            for line in txt:
                if len(line) > 30 and "]:" in line:
                    filename = line.split()[0]
                    content = line.split("]: ")[1]
                    if filename == temp_1[8].split('.')[0]:
                        # eg. filename：Ses01F_impro01_F000，在这里是要找到一个对话文件中跟train_filename_save中的utterance对应的utterance，并将其append到train_text中
                        train_text.append(content)
                        break
            f.close()
        i += 1
        if len(train_text) != i:
            train_text.append(" ")
        pass

    texts = [s.lower() for s in train_text]
    # 去除除了字母外的所有符号
    re_han = re.compile("([a-zA-Z]+)")

    train_text = []
    for line in texts:
        word = []
        # 将一个句子中的词分开
        blocks = re_han.split(line)
        for blk in blocks:
            if re_han.match(blk):
                word.extend(jieba.lcut(blk))
        train_text.append(word)

    train_label_in_num = [generate_label(e) for e in train_label]
    train_label_in_onehot = kr.utils.to_categorical(train_label_in_num, num_classes=5)

    f = open('./models/IEMOCAP_all.pkl', 'wb')
    pickle.dump((train_data, train_label, train_filename_save, train_text, train_label_in_num, train_label_in_onehot,
                 train_emt), f)
    f.close()
    pass


if __name__ == "__main__":
    read_IEMOCAP()