# -*- coding: utf-8 -*-
import nltk
import os
import os.path

from tqdm import tqdm

"""
Ubuntu Dialogue Corpus

http://arxiv.org/abs/1506.08909

"""

class UbuntuData:
    """
    """
    def __init__(self, dirName):
        """
        Args:
            dirName (string): data directory of ubuntu data
        """
        print('creating UbuntuData obj')

        self.conversations = []
        data_root_dir = os.path.join(dirName, 'dialogs')
        count = 0

        if os.path.isfile(os.path.join(data_root_dir, 'ubuntu3_7.pkl')):
            print('loading from ubuntu3_7.pkl')
            import pickle
            with open(os.path.join(data_root_dir, 'ubuntu3_7.pkl'),'rb') as f:
                self.conversations = pickle.load(f)
        else:
            for l in range(3, 8):
                data_dir = os.path.join(data_root_dir, str(l))
                for file in tqdm(os.listdir(data_dir)):
                    if file.endswith(".tsv"):
                        self.conversations.append(
                                {"lines":self.loadLines(os.path.join(data_dir, file))})
                print('there are %d %d-round conversations' % (len(self.conversations) - count, l))
                count = len(self.conversations)

    def loadLines(self, fileName):
        """
        Args:
            fileName (str): file to load
        Return:
            list<dict<str>>: the extracted fields for each line
        """
        lines = []
        with open(fileName, 'r') as f:
            lineID = 0
            for line in f:
                line = [x for x in line.split("\t") if x]
                speaker = line[1]
                content = line[-1].strip()
                if lineID==0 or lines[-1]["speaker"] != speaker:
                    lines.append({"speaker":speaker, "text": self.extractText(content)})
                else:
                    lines[-1]["text"] = lines[-1]["text"]+[["_eou_"]] + self.extractText(content)
                lineID += 1

        return lines

    def extractText(self, line):
        """从句子里面提取词语。
            Args:
                line (str): 一句话，将从中提取词语
            Return:
                list<list<int>>: the list of sentences of word ids of the sentence
        """
        sentences = []  # List[List[str]]
        ##print('input line: %s' % line)
        # Extract sentences
        sentencesToken = nltk.sent_tokenize(line)
        ##print('sent_tokenized sentencesToken: %s' % sentencesToken)
        """
        e.g.
        >> line = "this is one line, and let's read it"
        >> nltk.word_tokenize(line)
        ['this', 'is', 'one', 'line', ',', 'and', 'let', "'s", 'read', 'it']
        """
        # We add sentence by sentence until we reach the maximum length
        for i in range(len(sentencesToken)):
            tokens = nltk.word_tokenize(sentencesToken[i])
            ##print('%d: \n\tword_tokenized tokens: %s\t\nfrom %s' %
            ##(i, tokens, sentencesToken[i]))

            tempWords = []
            for token in tokens:
                if '/' in token: # url in token
                    token_words = []
                    for w in token.strip('/').split('/'):
                        if '.' in w:
                            token_words.extend([x for x in w.split('.') if x])
                        elif len(w)>0:
                            token_words.append(w)
                    tempWords.extend(token_words)
                else:
                    tempWords.append(token)
            sentences.append(tempWords)
        ##print('finally, sentences = %s' % sentences)

        return sentences


    def getConversations(self):
        return self.conversations

if __name__ == "__main__":
    dirName = '/home/dongguo/Dropbox/Projects/NLP/DeepQA/data/ubuntu/'
    data = UbuntuData(dirName)
    conversations = data.getConversations()
    import pickle
    with open('ubuntu3_7.pkl','wb') as f:
        pickle.dump(conversations, f)
