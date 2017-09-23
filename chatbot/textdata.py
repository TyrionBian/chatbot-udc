# -*- coding: utf-8 -*-

from tqdm import tqdm  # Progress bar

import collections
import nltk
import numpy as np
import os
import pickle
import random
import string

from corpus.cornelldata import CornellData
from corpus.ubuntudata import UbuntuData
from corpus.xhjdata import XhjData


def tqdm_wrap(iterable, *args, **kwargs):
    """Forward an iterable eventually wrapped around a tqdm decorator
    The iterable is only wrapped if the iterable contains enough elements
    Args:
        iterable (list): An iterable object which define the __len__ method
        *args, **kwargs: the tqdm parameters
    Return:
        iter: The iterable eventually decorated
    """
    if len(iterable) > 100:
        return tqdm(iterable, *args, **kwargs)
    return iterable


class Batch:
    """Struct containing batches info
    """
    def __init__(self):
        # nice summary of various sequence required by chatbot training.
        self.query_seqs = []
        self.response_seqs = []
        # self.xxlength：encoding阶段调用 dynamic_rnn 时的一个 input_argument
        self.query_length = []
        self.response_length = []


class TextData:
    """chat数据处理
    """

    # OrderedDict because the first element is the default choice
    availableCorpus = collections.OrderedDict([
        ('ubuntu', UbuntuData),
        ('cornell', CornellData),
        ('xiaohuangji', XhjData)
    ])

    @staticmethod
    def corpusChoices():
        """Return the dataset availables
        Return:
            list<string>: the supported corpus
        """
        return list(TextData.availableCorpus.keys())

    def __init__(self, args):
        """读入/处理 所有的对话数据。
        Args:
            args: 模型的超参数
        """
        self.args = args

        # 指定保存数据集（包含词典）的路径
        self._processPaths()

        # 用于保存数据信息的变量和数据结构
        # 1. 几个常用的特殊符号
        self.padToken = -1  # Padding，用来将一些不同长度的句子补齐成相同长度的 minibatch
        self.goToken = -1   # Start of sequence，用作句子的开头
        self.eosToken = -1  # End of sequence，标记句子的结尾
        self.unknownToken = -1  # Word dropped from vocabulary，表示词典外的词语
        # 2. 词典
        self.word2id = {} # 词典，{单词：int}
        self.id2word = {} # 词典，{int：单词}
        self.idCount = {} # 词频统计用的dict，{单词：count},用来过滤稀有词汇
        # 3. 数据存储在一个或者三个list里面
        # train/valid/test: list<[input,target]>，包含（提问，回复）格式的样本
        #   提问input 和 回复target都是 list<int>
        self.trainingSamples = []
        self.validationSamples = []
        self.testingSamples = []

        # 读入处理后的数据集，如果不存在，则处理和保存数据集
        self.loadCorpus()

        # 打印一些数据信息：
        self._printStats()

    def _processPaths(self):
        """处理模型和数据相关的路径，命名问题

        例如：给定数据地址类超参数
        （args.rootDir，args.corpus， args.datasetTag) = ('lecture3', 'ubuntu', 'round3_5')
        以及数据处理类超参数
        （args.maxLength, args.filterVocab, args.vocabularySize）= （20， 1， 20K）
        self.corpusDir = 'lecture3/data/ubuntu'
        self.fullSamplesPath = 'lecture3/data/samples/dataset-ubuntu-round3_5.pkl'
        self.filteredSamplesPath = 'lecture3/data/samples/dataset-ubuntu-round3_5-length20-filter1-vocaSize20000.pkl'
        """

        self.corpusDir = os.path.join(
            self.args.rootDir, 'data', self.args.corpus)

        targetPath = os.path.join(self.args.rootDir, 'data/samples',
                                 'dataset-{}'.format(self.args.corpus))
        if self.args.datasetTag:
            targetPath += '-' + self.args.datasetTag

        # 完整（原始）数据集的地址和文件名
        self.fullSamplesPath = targetPath + '.pkl'

        # 处理过后的数据集的地址和文件名
        # 保存在文件名中的三个数据处理类的超参数是：
        #    maxLength： 对话的每一轮长度不超过maxLength
        #    filterVocab: 出现频率低于filterVocab的词语使用UNK替换掉
        #    vocabularySize： 除掉频率低于filterVocab的词语以后，出现频率最高的vocabularySize个单词留下
        self.filteredSamplesPath = targetPath + '-length{}-filter{}-vocabSize{}.pkl'.format(
            self.args.maxLength,
            self.args.filterVocab,
            self.args.vocabularySize,
        )


    def _printStats(self):
        print('Loaded {}: {} words, {} training QA samples'.format(
            self.args.corpus, len(self.word2id), len(self.trainingSamples)))


    def shuffle(self):
        """Shuffle the training samples
        """
        print('Shuffling the dataset...')
        random.shuffle(self.trainingSamples)


    ## =======================================================================================
    ## 从/向 外界 读/创建+写 文件
    # 1. 主要函数，从指定位置读数据集，如果不存在，则创建数据集
    def loadCorpus(self):
        """读/创建 对话数据：
        在训练文件创建的过程中，由两个文件
            1. self.fullSamplePath
            2. self.filteredSamplesPath
        """
        print('filteredSamplesPath:%s' % self.filteredSamplesPath)
        datasetExist = os.path.isfile(self.filteredSamplesPath)
        # 如果处理过的对话数据文件不存在，创建数据文件
        if not datasetExist:
            print('训练样本不存在。从原始样本数据集创建训练样本...')

            # 创建/读取原始对话样本数据集： self.trainingSamples
            print('fullSamplesPath:%s' % self.fullSamplesPath)
            datasetExist = os.path.isfile(self.fullSamplesPath)
            if not datasetExist:
                print('原始训练样本不存在。创建原始样本数据集...')
                # 1. 创建 corpus 对象, 例如 UDC corpus对象
                print('self.corpusDir: %s' % self.corpusDir)
                print('self.args.corpus: %s' % self.args.corpus)
                corpusData = TextData.availableCorpus[self.args.corpus](self.corpusDir)
                print(repr(corpusData))
                # 2. 读取和预处理数据，提取对话样本：
                self.createFullCorpus(corpusData.getConversations())
                # 3. 保存简单预处理后的原始训练数据集
                self.saveDataset(self.fullSamplesPath)
            else:
                self.loadDataset(self.fullSamplesPath)
            self._printStats()

            # 后续处理
            # 1. 单词过滤，去掉不常见(<=filterVocab)的单词，保留最常见的vocabSize个单词
            print('Filtering words (vocabSize = {} and wordCount > {})...'.format(
                self.args.vocabularySize,
                self.args.filterVocab
            ))
            self.filterFromFull()

            # 2. 分割数据
            print('分割数据为 train, valid, test 数据集...')
            n_samples = len(self.trainingSamples)
            train_size = int(self.args.train_frac * n_samples)
            valid_size = int(self.args.valid_frac * n_samples)
            test_size = n_samples - train_size - valid_size

            print('n_samples=%d, train-size=%d, valid_size=%d, test_size=%d' % (
                n_samples, train_size, valid_size, test_size))
            self.shuffle()
            self.testingSamples = self.trainingSamples[-test_size:]
            self.validationSamples = self.trainingSamples[-valid_size-test_size : -test_size]
            self.trainingSamples = self.trainingSamples[:train_size]

            # 保存处理过的训练数据集
            print('Saving dataset...')
            self.saveDataset(self.filteredSamplesPath)
        else:
            self.loadDataset(self.filteredSamplesPath)

        assert self.padToken == 0

    # 2. utility 函数，使用pickle写文件
    def saveDataset(self, filename):
        """使用pickle保存数据文件。

        数据文件包含词典和对话样本。

        Args:
            filename (str): pickle 文件名
        """
        with open(filename, 'wb') as handle:
            data = {
                    'word2id': self.word2id,
                    'id2word': self.id2word,
                    'idCount': self.idCount,
                    'trainingSamples': self.trainingSamples
            }

            if len(self.validationSamples)>0:
                data['validationSamples'] = self.validationSamples
                data['testingSamples'] = self.testingSamples

            pickle.dump(data, handle, -1)  # Using the highest protocol available


    # 3. utility 函数，使用pickle读文件
    def loadDataset(self, filename):
        """使用pickle读入数据文件
        Args:
            filename (str): pickle filename
        """
        dataset_path = os.path.join(filename)
        print('Loading dataset from {}'.format(dataset_path))
        with open(dataset_path, 'rb') as handle:
            data = pickle.load(handle)
            self.word2id = data['word2id']
            self.id2word = data['id2word']
            self.idCount = data.get('idCount', None)
            self.trainingSamples = data['trainingSamples']
            if 'validationSamples' in data:
                self.validationSamples = data['validationSamples']
                self.testingSamples = data['testingSamples']

            self.padToken = self.word2id['<pad>']
            self.goToken = self.word2id['<go>']
            self.eosToken = self.word2id['<eos>']
            self.unknownToken = self.word2id['<unknown>']


    ## =======================================================================================
    ## 在本地内存处理数据 (corpus)
    # 1. utility 函数，产生/使用 词典
    def getWordId(self, word, create=True):
        """返回单词的整数ID，并在遇到新单词的时候更新词典。

        用于将完整文字数据集转化为完整的整数格式数据集.

        如果单词不在词典之中，
        1. 如果create=False，则将单词记录为<UNK>
        2. 反之，则将其添加到词典里面

        Args:
            word (str): 要返回ID的单词
            create (Bool): 如果 True 并且是一个新词，将新词添加到辞典中
        Return:
            int: 单词的ID

        """

        word = word.lower()

        # 在推理(inference)/测试(test, evaluation)阶段, create设置成False
        if not create:
            wordId = self.word2id.get(word, self.unknownToken)
        elif word in self.word2id:
            wordId = self.word2id[word]
            self.idCount[wordId] += 1
        else:
            # 需要一些词语的筛选操作？
            wordId = len(self.word2id)
            self.word2id[word] = wordId
            self.id2word[wordId] = word
            self.idCount[wordId] = 1

        return wordId


    # 2. utility 函数，对话==>句子
    def extractConversation(self, conversation):
        """从一个对话样本中，提取句子

        Args:
            conversation (Obj): 一个 conversation 对象，包含对话内容
        """

        if self.args.skipLines:
            # WARNING: The dataset won't be regenerated if the choice evolve
            # (have to use the datasetTag)
            step = 2
        else:
            step = 1

        # 一个N轮对话里面，每一行和下一行组成一个单轮对话样本
        # 如果 skipLines，则每一行不会被重复利用
        # 反之，第i行是第(i-1)个对话的 “答话” 和第i个对话的 “问话”
        for i in tqdm_wrap(
            range(0, len(conversation['lines']) - 1, step),
            desc='Conversation',
            leave=False
        ):
            inputLine  = conversation['lines'][i]['text']
            targetLine = conversation['lines'][i+1]['text']

            inputWords  = []
            targetWords = []

            try:
                for i in range(len(inputLine)):
                    if len(inputLine[i])>0:
                        inputWords.append([])
                        for j in range(len(inputLine[i])):
                            inputWords[-1].append(self.getWordId(inputLine[i][j]))

                for i in range(len(targetLine)):
                    if len(targetLine[i])>0:
                        targetWords.append([])
                        for j in range(len(targetLine[i])):
                            targetWords[-1].append(self.getWordId(targetLine[i][j]))

            except:
                print('inputWords %s' % inputLine['text'])
                print('targetWords %s' % targetLine['text'])
            # 过滤掉空的对话
            if inputWords and targetWords:
                self.trainingSamples.append([inputWords, targetWords])


    # 4. 主要函数，创建完整的原始数据集
    def createFullCorpus(self, conversations):
        """创建原始的对话数据集，并创建无字数限制的词典

        注意整个数据集会被预处理，但是不限制句子长度或者词典大小。
        """
        # 先指定特殊词语
        self.padToken = self.getWordId('<pad>')  # Padding
        self.goToken = self.getWordId('<go>')    # Start of sequence
        self.eosToken = self.getWordId('<eos>')  # End of sequence
        self.unknownToken = self.getWordId('<unknown>')  # UNK Word dropped from vocabulary

        # 预处理，提取对话
        for conversation in tqdm(conversations, desc='Extract conversations'):
            self.extractConversation(conversation)

    # 5. 主要函数，过滤数据
    def filterFromFull(self):
        """ 读入预处理过的原始数据集，根据超参数设置过滤单词和句子
        """

        def mergeSentences(sentences, fromEnd=False):
            """拼接句子，过滤掉超过超参数允许的句子长度的部分。并在过滤的同时更新词频统计。

            Args:
                sentences (list<list<int>>): 一个list，其中每一个元素是list<int>格式的句子
                fromEnd (bool): 一个比较tricky的技巧，要达到如下效果：
                    如果作为“问句”的长度太长的话，保留问句的后半部分；
                    如果作为“答句”的长度太常的话，保留答句的前半部分。
            Return:
                list<int>: list<int> 格式的句子
            """
            merged = []

            if fromEnd:
                sentences = reversed(sentences)

            for sentence in sentences:

                # 如果没有超过允许的最长句子长度，继续拼接短句。
                if len(merged) + len(sentence) <= self.args.maxLength:
                    if fromEnd:  # Append the sentence
                        merged = sentence + merged
                    else:
                        merged = merged + sentence
                else:
                    # 用于在处理过程中更新词典：如果一些句子太长而被过滤掉的话，在词频统计中
                    # 减去相应的出现次数
                    try:
                        for w in sentence:
                            self.idCount[w] -= 1
                    except:
                        print('sentence: %s' % sentence)
            return merged

        newSamples = []

        # 第一步，过滤句子，去掉过长的部分
        for inputWords, targetWords in tqdm(
            self.trainingSamples, desc='Filter sentences:', leave=False):

            try:
                inputWords = mergeSentences(inputWords, fromEnd=True)
            except:
                print('inputWords: %s' % inputWords)
            targetWords = mergeSentences(targetWords, fromEnd=False)
            newSamples.append([inputWords, targetWords])

        # 第二步，过滤掉 unused words and replace them by the unknown token, 并且相应地更新字典
        # unused words 是什么？

        # 先选择词汇表：包括特殊词汇和出现频率最高的vocabularySize个词语。词频统计是过滤过句子以后的版本。
        specialTokens = {
            self.padToken,
            self.goToken,
            self.eosToken,
            self.unknownToken
        }
        selectedWordIds = collections.Counter(self.idCount).most_common(
            self.args.vocabularySize or None)  # Keep all if vocabularySize == 0
        selectedWordIds = {k for k, v in selectedWordIds if v > self.args.filterVocab}
        selectedWordIds = selectedWordIds.union(specialTokens)


        newMapping = {}  # Map the full words ids to the new one (TODO: Should be a list)
        newId = 0
        for wordId, count in [(i, self.idCount[i]) for i in range(len(self.idCount))]:
            # wordId, count: 在旧的词典里面的单词ID和词频。
            if wordId in selectedWordIds:  # Update the word id
                word = self.id2word[wordId]
                # 更新(word, wordId)
                # 1. wordId 更新为 newId
                newMapping[wordId] = newId
                # 2. newId <= wordId
                # 2-a. 更新 word = id2word[wordId] ==>  id2word[newId]
                # 2-b. 更新 word2id[word] = wordId ==> newId
                del self.id2word[wordId]
                self.word2id[word] = newId
                self.id2word[newId] = word
                newId += 1
            else:
                # 删除原来的辞典中未被选进selectedWordIds的词语，将其更改为UNK；
                # 我们不更新词频统计，因为以后用不到了。
                newMapping[wordId] = self.unknownToken
                del self.word2id[self.id2word[wordId]]  # The word isn't used anymore
                del self.id2word[wordId]

        # 最后一步，更新词语ID，删除空的句子
        # 如果一句话里面全都是UNK，则valid==False,所属的整个对话样本将被删去
        def replace_words(words):
            valid = False  # Filter empty sequences
            for i, w in enumerate(words):
                words[i] = newMapping[w]
                if words[i] != self.unknownToken:  # Also filter if only contains unknown tokens
                    valid = True
            return valid

        del self.trainingSamples[:]

        for inputWords, targetWords in tqdm(newSamples, desc='Replace ids:', leave=False):
            valid = True
            valid &= replace_words(inputWords)
            valid &= replace_words(targetWords)
            valid &= targetWords.count(self.unknownToken) == 0
            # 如果答句（target Words）中包含unk,则丢弃这个对话样本

            if valid:
                self.trainingSamples.append([inputWords, targetWords])  # TODO: Could replace list by tuple

        self.idCount.clear()


    ## ==================================================================================
    ## 与 model running 相关的函数，用于产生minibatch数据样本
    def _createBatch(self, samples):
        """从输入的一个样本的list构建一个batch. 输入数据的样本数自动确定batch size.

        输入数据中的句子应该已经被逆序过。The inputs should already be **inverted**. (??? really? there are codes that revert the input, i.e. `reversed(sample[0])`)
        输出数据中的句子应该在开头和结尾包含 <go> 和 <eos>.

        Warning: 这个函数不应该直接调用args.batchSize !!!

        Args:
            samples (list<Obj>): 样本的list, 每个样本是 [input, target] 的格式

        Return:
            Batch: 一个 batch 对象

        TODO：
            产生一些DEMO代码，打印一个batch的格式和内容。
        """

        batch = Batch()
        batchSize = len(samples)

        # Create the batch tensor
        for i in range(batchSize):
            # Unpack the sample
            sample = samples[i]

            batch.query_seqs.append(sample[0])
            batch.response_seqs.append(sample[1])
            batch.query_length.append(len(batch.query_seqs[-1]))
            batch.response_length.append(len(batch.response_seqs[-1]))

            # Long sentences should have been filtered during the dataset creation
            assert len(batch.query_seqs[i]) <= self.args.maxLength
            assert len(batch.response_seqs[i]) <= self.args.maxLength

            # fill with padding to align batchSize samples into one 2D list
            batch.query_seqs[i] = batch.query_seqs[i] + [self.padToken] * (self.args.maxLength - len(batch.query_seqs[i]))
            batch.response_seqs[i]  = batch.response_seqs[i]  + [self.padToken] * (self.args.maxLength - len(batch.response_seqs[i]))

        return batch


    def getBatches(self):
        """Prepare the batches for the current epoch
        Return:
            list<Batch>: Get a list of the batches for the next epoch
        """
        self.shuffle()

        batches = []

        def genNextSamples():
            """ Generator over the mini-batch training samples
            """
            for i in range(0, self.getSampleSize(), self.args.batchSize):
                yield self.trainingSamples[i:min(i + self.args.batchSize, self.getSampleSize())]

        for samples in genNextSamples():
            batch = self._createBatch(samples)
            batches.append(batch)
        return batches


    ## ==================================================================================
    ## 一系列数据转换方法,用于具体样本，和整个数据集的操作无关
    # 1. list<int> ==> list<str> ==> str (句子）
    def sequence2str(self, sequence, clean=False, reverse=False):
        """翻译：将一列(list)整数 转化为可读的字符串
        Args:
            sequence (list<int>): 一列整数
            clean (Bool): 如果True，去掉 <go>, <pad> 和 <eos>
            reverse (Bool): 适用于input，是否恢复正常顺序
        Return:
            str: 字符串，内容是可读的句子
        """

        if not sequence:
            return ''

        if not clean:
            sentence = [self.id2word[idx] for idx in sequence]

        sentence = []
        for wordId in sequence:
            if wordId == self.eosToken:  # End of generated sentence
                break
            elif wordId != self.padToken and wordId != self.goToken:
                sentence.append(self.id2word[wordId])

        if reverse:  # Reverse means input so no <eos> (otherwise pb with previous early stop)
            sentence.reverse()

        def detokenize(self, tokens):
            """自定义的 ' '.join() 方法，对标点符号特殊处理
                如果是标点符号，则不在前面添加空格；
                如果单引号（适用于英文，例如[Let 's], [some_name 's]），则不在前面添加空格；
                否则，则在前面添加一个空格。
            Args:
                tokens (list<string>): 一列（list）字符串
            Return:
                str: 拼到一起的句子
            """
            return ''.join([
                ' ' + t if not t.startswith('\'') and
                           t not in string.punctuation
                        else t
                for t in tokens]).strip().capitalize()

        return self.detokenize(sentence)


    # 2. list<list<int>> ==> list<list<str>> ==> list<str>
    def batchSeq2str(self, batchSeq, seqId=0, **kwargs):
        """翻译：将batch_size列(list)整数 转化为batch_size句可读的字符串

        Args:
            batchSeq (list<list<int>>): 需要被翻译成可读字符串的 batch_size 列(list)整数
            seqId (int): 将要处理 batch 中的每个样本中的第 seqId 个 sequence 将要被处理
            kwargs: 格式选项( 见 sequence2str() )
        Return:
            str: batch句可读的字符串
        """
        sequence = []
        for i in range(len(batchSeq)):  # Sequence length
            sequence.append(batchSeq[i][seqId])
        return self.sequence2str(sequence, **kwargs)


    # 3.
    def sentence2enco(self, sentence):
        """Encode a sequence and return a batch as an input for the model
        Return:
            Batch: a batch object containing the sentence, or none if something went wrong
        """

        if sentence == '':
            return None

        # 第一步，分词
        tokens = nltk.word_tokenize(sentence)
        if len(tokens) > self.args.maxLength:
            return None

        # 第二步，将单词转化为整数ID
        wordIds = []
        for token in tokens:
            wordIds.append(self.getWordId(token, create=False))

        # 第三部，创建batch（包含padding，逆序等操作）
        batch = self._createBatch([[wordIds, []]])  # Mono batch, no target output

        return batch


    # 4.
    def deco2sentence(self, decoderOutputs):
        """Decode the output of the decoder and return a human friendly sentence
        decoderOutputs (list<np.array>):
        """
        sequence = []

        # Choose the words with the highest prediction score
        for out in decoderOutputs:
            sequence.append(np.argmax(out))  # Adding each predicted word ids

        return sequence  # We return the raw sentence. Let the caller do some cleaning eventually


    ## ==================================================================================
    ## 其他
    def playDataset(self):
        """Print a random dialogue from the dataset
        """
        print('Randomly play samples:')
        for i in range(self.args.playDataset):
            idSample = random.randint(0, len(self.trainingSamples) - 1)
            print('Q: {}'.format(self.sequence2str(self.trainingSamples[idSample][0], clean=True)))
            print('A: {}'.format(self.sequence2str(self.trainingSamples[idSample][1], clean=True)))
            print()
        pass


    #TODO: change to decoratoer
    def getSampleSize(self):
        """Return the size of the dataset
        Return:
            int: Number of training samples
        """
        return len(self.trainingSamples)


    def getVocabularySize(self):
        """Return the number of words present in the dataset
        Return:
            int: Number of word on the loader corpus
        """
        return len(self.word2id)


    def printBatch(self, batch):
        """打印一个batch，用于debug
        Args:
            batch (Batch): 一个batch对象
        """
        print('----- Print batch -----')
        for i in range(len(batch.query_seqs[0])):  # Batch size
            print('Encoder: {}'.format(self.batchSeq2str(batch.query_seqs, seqId=i)))
            print('Targets: {}'.format(self.batchSeq2str(batch.response_seqs, seqId=i)))


class RankTextData(TextData):

    def __init__(self, args):
        """Load all conversations
        Args:
            args: parameters of the model
        """
        # 调用base class的初始化程序，读入/创建+读入 args指定的数据集
        super(RankTextData, self).__init__(args)
        del self.trainingSamples[:]
        self.valid_neg_idx = []
        self.test_neg_idx = []

    def _sampleNegative(self):
        """ 对于validation和testing数据集中的每一个样本，产生19个negative回复。

        用于计算 Recall@k。

        """
        def npSampler(n):
            """产生 [n x 19] 的np.array，第i行表示第i个输入对应的19个固定的错误回复的indices。
            """
            neg = np.zeros(shape = (n, 19))
            for i in range(19):
                neg[:,i] = np.arange(n)
                np.random.shuffle(neg[:,i])
            findself = neg - np.arange(n).reshape([n, 1])
            findzero = np.where(findself==0)
            for (r, c) in zip(findzero[0], findzero[1]):
                x = np.random.randint(n)
                while x == r:
                    x = np.random.randint(n)
                neg[r, c] = x
            return neg.astype(int)

        n_valid = len(self.validationSamples)
        ##print('generating valid_neg_idx for %d' % n_valid)
        self.valid_neg_idx = npSampler(n_valid)
        n_test = len(self.testingSamples)
        ##print('generating test_neg_idx for %d' % n_test)
        self.test_neg_idx = npSampler(n_test)


    def _createEvalBatch(self, samples, dataset, neg_responses):
        """从输入的一个样本的list构建一个batch.
        batch_size由输入数据的样本数自动确定，将被输入dynamic_rnn计算thought vector.

        Args:
            samples (list<Obj>): 样本的list, 每个样本是 [input, target] 的格式
            dataset: self.validationSamples 或者 self.testingSamples
            neg_responses (2D array): neg_responses[i][j] 表示samples中的第i个样本的第j个错误回复

        Return:
            Batch: 一个 batch 对象

        TODO：
            产生一些DEMO代码，打印一个batch的格式和内容。
        """

        batch = Batch()
        batchSize = len(samples)

        # Create the batch tensor
        # using dynamic_rnn
        # time_major == False (default), shape = [batch_size, max_time, ...]

        for i in range(batchSize):

            sample = samples[i]

            batch.query_seqs.append(sample[0])
            batch.query_length.append(len(sample[0]))
            assert batch.query_length[-1] <= self.args.maxLength

            batch.response_seqs.append(sample[1])
            batch.response_length.append(len(sample[1]))
            assert batch.response_length[-1] <= self.args.maxLength

            for j in range(19):
                sample = dataset[neg_responses[i][j]]
                batch.response_seqs.append(sample[1])
                batch.response_length.append(len(sample[1]))
                assert batch.response_length[-1] <= self.args.maxLength

            # pad句子到同样长度，从而将一个minibatch的样本存进一个tensor里面
            batch.query_seqs[i] = batch.query_seqs[i] + [self.padToken] * (
                self.args.maxLength  - len(batch.query_seqs[i]))

            for j in range(20):
                batch.response_seqs[i*20 + j] = batch.response_seqs[i*20 + j] + [self.padToken] * (
                    self.args.maxLength - len(batch.response_seqs[i*20 + j]))

        return batch


    def getValidBatches(self):
        """产生用于衡量模型的一个minibatch

        每一个输入样本，对应1个真实的回复样本和19个随机回复作为错误回复样本

        Return:
            batches: list<Batch>
        """

        # 如果valid_neg_idx为空，即，还没有对每个valid样本产生错误回复数据，调用一次_sampleNegative
        if self.valid_neg_idx == []:
            self._sampleNegative()

        batches = []
        n_valid = len(self.validationSamples)

        def genNextSamples():
            """ Generator over the mini-batch training samples
            """
            for i in range(0, n_valid, self.args.batchSize):
                yield (self.validationSamples[i:min(i + self.args.batchSize, n_valid)],
                        self.valid_neg_idx[i:min(i + self.args.batchSize, n_valid), :])

        for (samples, neg_responses) in genNextSamples():
            batch = self._createEvalBatch(samples, self.validationSamples, self.valid_neg_idx)
            batches.append(batch)

        return batches

    def getTestBatches(self):
        """功能等同于getValidBatches，唯一区别是作用于testing data
        """

        # 如果test_neg_idx为空，即，还没有对每个valid样本产生错误回复数据，调用一次_sampleNegative
        if self.test_neg_idx == []:
            self._sampleNegative()


        batches = []
        n_test = len(self.testingSamples)

        def genNextSamples():
            """ Generator over the mini-batch training samples
            """
            for i in range(0, n_test, self.args.batchSize):
                yield (self.testingSamples[i:min(i + self.args.batchSize, n_test)],
                        self.test_neg_idx[i:min(i + self.args.batchSize, n_test), :])

        for (samples, neg_responses) in genNextSamples():
            batch = self._createEvalBatch(samples, self.testingSamples, self.test_neg_idx)
            batches.append(batch)
        return batches
