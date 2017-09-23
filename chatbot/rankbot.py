"""
Retrieval and ranking base chatbot. Referred to DeepQA repo.

Use python 3

"""

import argparse # 用于分析输入的超参数
import configparser  # 保存模型的超参数到一个可读文件里面
import datetime  # 训练过程记时
import os  # 文件管理
import tensorflow as tf
import time

from tqdm import tqdm  # 进度条
from textdata import (TextData, RankTextData)
from ranker import Ranker

class Rankbot:
    """
    Retrieval-based chatbot
    """

    def __init__(self):

        self.args = None
        self.textData = None
        self.model = None

        self.modelDir = ''
        self.globStep = 0
        self.ckpt_model_saver = None
        self.best_model_saver = None
        self.best_model = []
        self.best_valid_loss = [float('inf'),
                                float('inf'),
                                float('inf')]

        self.sess = None

        self.MODEL_DIR_BASE = 'save/model'
        self.MODEL_NAME_BASE = 'model'
        self.BEST_MODEL_NAME_BASE = 'best_model'
        self.MODEL_EXT = '.ckpt'
        self.CONFIG_FILENAME = 'params.ini'


    @staticmethod
    def parseArgs(args):
        """
        Parse 超参数
        Args:
            args (list<stir>): List of arguments.
        """

        parser = argparse.ArgumentParser()

        # Global options
        globalArgs = parser.add_argument_group('Global选项')
        globalArgs.add_argument('--keepAll', action='store_true',
                                help='如果等于True，则保留所有的中间结果')
        globalArgs.add_argument('--modelTag', type=str, default=None,
                                help='模型的标记，方便以后识别，区分和管理不同的模型')
        globalArgs.add_argument('--rootDir', type=str, default=None,
                                help='保存模型和数据的根目录')
        globalArgs.add_argument('--device', type=str, default=None,
                                help='\'gpu\' or \'cpu\'，指定运算用的设备')
        globalArgs.add_argument('--seed', type=int, default=None,
                                help='随机数种子，方便重现实验结果')

        # 数据相关的选项
        datasetArgs = parser.add_argument_group('数据处理超参数')
        datasetArgs.add_argument('--corpus', choices=TextData.corpusChoices(),
                                 default=TextData.corpusChoices()[0],
                                 help='数据集选项.')
        datasetArgs.add_argument('--datasetTag', type=str, default='',
                                 help='数据集的标记，方便数据的版本控制。例如，'
                                 '我们产生一个20000个单词的数据文件和另一个40000个单词的数据文件')
        datasetArgs.add_argument('--maxLength', type=int, default=10,
                                 help='输入/问，输出/答句子的最长长度，对应RNN的最长长度')
        datasetArgs.add_argument('--filterVocab', type=int, default=1,
                                 help='去掉出现频率 <= filterVocab的词语。若要保留所有单词，filterVocab设置为0')
        datasetArgs.add_argument('--skipLines', action='store_true',
                                 help='如果等于True，只使用对话记录中的[2*i, 2*i+1]行作为数据样本'
                                 '否则，[2*i, 2*i+1]也可以作为数据样本，'
                                 '对话纪录中出去第一行和最后一行的每一行会出现在两个样本里面')
        datasetArgs.add_argument('--vocabularySize', type=int, default=20000,
                                 help='词典大小的上限(0 表示没有上限)')
        datasetArgs.add_argument('--train_frac', type=float, default = 0.8,
                                 help ='percentage of training samples')
        datasetArgs.add_argument('--valid_frac', type=float, default = 0.1,
                                 help ='percentage of training samples')

        # 模型结构选项
        nnArgs = parser.add_argument_group('结构相关的模型超参数')
        nnArgs.add_argument('--hiddenSize', type=int, default=256,
                            help='每个RNN cell的state维度')
        nnArgs.add_argument('--numLayers', type=int, default=2,
                            help='每个时间的RNN cell层数')
        nnArgs.add_argument('--initEmbeddings', action='store_true',
                            help='如果True, 使用开源的 word2vec 参数初始化词向量')
        nnArgs.add_argument('--embeddingSize', type=int, default=256,
                            help='词向量维度')
        nnArgs.add_argument('--embeddingSource', type=str,
                            default="GoogleNews-vectors-negative300.bin",
                            help='用来初始化词向量的 word2vec 文件')

        # 模型训练设置
        trainingArgs = parser.add_argument_group('Training options')
        trainingArgs.add_argument('--numEpochs', type=int, default=25,
                                  help='设置训练多少个epoch')
        trainingArgs.add_argument('--saveEvery', type=int, default=5000,
                                  help='设置经过多少个minibatch记录一次checkpoint')
        trainingArgs.add_argument('--batchSize', type=int, default=32,
                                  help='mini-batch样本数量')
        trainingArgs.add_argument('--learningRate', type=float, default=0.002,
                                  help='Learning rate')
        trainingArgs.add_argument('--dropout', type=float, default=0.9,
                                  help='Dropout rate (这里是dropout以后保留的比重,keep_prob)')

        return parser.parse_args(args)


    def main(self, args=None):
        """训练bot的主函数
        """
        print('Welcome to DeepRank!')
        print()
        print('TensorFlow detected: v{}'.format(tf.__version__))

        # 初始化, hyperparameters
        self.args = self.parseArgs(args)
        if not self.args.rootDir:
            self.args.rootDir = os.getcwd()
        self.loadHyperParams()

        # 读入训练和测试用的数据
        self.textData = TextData(self.args)
        self.evalData = RankTextData(self.args)

        # 搭建模型
        graph = tf.Graph()
        with tf.device(self.getDevice()):
            with graph.as_default():
                with tf.name_scope('training'):
                        self.model_train = Ranker(self.args, is_training = True)

                tf.get_variable_scope().reuse_variables()
                with tf.name_scope('validation'):
                    self.model_valid = Ranker(self.args, is_training = False)

                with tf.name_scope('evluation'):
                    self.model_test = Ranker(self.args, is_training = False)
                    self.ckpt_model_saver = tf.train.Saver(name = 'checkpoint_model_saver')
                    self.best_model_saver = tf.train.Saver(name = 'best_model_saver')


                # Running session
                # allow_soft_placement = True: 当设置为使用GPU而实际上没有GPU的时候，允许使用其他设备运行。
                self.sess = tf.Session(config=tf.ConfigProto(
                    allow_soft_placement=True,
                    log_device_placement=False)
                )
                print('Initialize variables...')
                self.sess.run(tf.global_variables_initializer())

                # 定义 saver/summaries
                graph_info = self.sess.graph
                self.train_writer = tf.summary.FileWriter(os.path.join(self.modelDir, 'train/'), graph_info)
                self.valid_writer = tf.summary.FileWriter(os.path.join(self.modelDir, 'valid/'), graph_info)

                """
                # 使用worvecd等预处理的词向量参数初始化 bot 模型的词向量参数
                if self.args.initEmbeddings:
                    self.loadEmbedding(self.sess)
                """

                # 开始训练
                self.mainTrain(self.sess)


    def mainTrain(self, sess):
        """ 训练模型
        Args:
            sess: 当前的 tf session
        """

        print('开始训练模型，（按 Ctrl+C 保存并推出训练过程）...')

        try:
            batches_valid = self.evalData.getValidBatches()
            batches_test = self.evalData.getTestBatches()
            for e in range(self.args.numEpochs):
                print()
                print("----- Epoch {}/{} ; (lr={}) -----".format(
                    e+1, self.args.numEpochs, self.args.learningRate))

                batches = self.textData.getBatches()

                tic = datetime.datetime.now()
                for nextBatch in tqdm(batches, desc="Training"):
                    # Training pass
                    ops, feedDict = self.model_train.step(nextBatch)
                    assert len(ops) == 3  # (training, loss)
                    _, loss, train_summaries = sess.run(ops, feedDict)
                    self.globStep += 1

                    # 记录训练状态（训练数据上的损失函数）
                    if self.globStep % 100 == 0:
                        tqdm.write("----- Step %d -- CE Loss %.2f" % (self.globStep, loss))

                    # Checkpoint
                    if self.globStep % self.args.saveEvery == 0:
                        self.train_writer.add_summary(train_summaries, self.globStep)
                        self.train_writer.flush()

                        # validation pass
                        print('Evaluating on validation data ...')
                        self.valid_losses = [0,0,0]
                        for nextEvalBatch in tqdm(batches_valid, desc="Validation"):
                            ops, feedDict = self.model_valid.step(nextEvalBatch)
                            assert len(ops)==2
                            loss, eval_summaries = sess.run(ops, feedDict)
                            for i in range(3):
                                self.valid_losses[i] += loss[i]

                        self.valid_writer.add_summary(eval_summaries, self.globStep)
                        self.valid_writer.flush()

                        for i in range(3):
                            self.valid_losses[i] = self.valid_losses[i]/len(batches_valid)

                        print('validation, Recall_20@(1,3,5) = %s' % self.valid_losses)
                        time.sleep(5)
                        if (len(self.best_model)==0) or (self.valid_losses[0] > self.best_valid_loss[0]):
                            print('best_model updated, with best accuracy :%s' % self.valid_losses)
                            self.best_valid_loss = self.valid_losses[:]
                            self._saveBestSession(sess)

                        self._saveCkptSession(sess)

                toc = datetime.datetime.now()
                print("Epoch %d finished in %s seconds" % (e, toc-tic))

            # 训练结束后，在测试数据上运行一遍
            self.best_model_saver.restore(sess, self.best_model)
            self.test_losses = [0,0,0]
            for nextTestBatch in tqdm(batches_test, desc = "FinalTest"):
                ops, feedDict = self.model_test.step(nextTestBatch)
                assert len(ops)==2
                loss, _ = sess.run(ops, feedDict)
                for i in range(3):
                    self.test_losses[i] += loss[i]/len(batches_test)
            print('Final testing, Recall_20@(1,3,5) = %s' % self.test_losses)


        except (KeyboardInterrupt, SystemExit):
            # 如果用户在程序运行过程中按 Ctrl+C 结束训练
            print('Interruption detected, exiting the program...')

        self._saveCkptSession(sess)  # Ultimate saving before complete exit


    def _saveCkptSession(self, sess):
        """ 保存模型参数

        Args:
            sess: 当前的tf session
        """
        tqdm.write('保存Checkpoint (don\'t stop the run)...')
        tqdm.write('validation, Recall_20@(1,3,5) = ' + repr(self.valid_losses))
        self.saveHyperParams()

        # 保存 checkpoint 的文件名
        model_name = os.path.join(self.modelDir, self.MODEL_NAME_BASE)
        if self.args.keepAll:
            model_name += '-' + str(self.globStep)
        model_name = model_name + self.MODEL_EXT

        self.ckpt_model_saver.save(sess, model_name)
        tqdm.write('Checkpoint saved.')


    def _saveBestSession(self, sess):
        """ 保存模型参数

        Args:
            sess: 当前的tf session
        """
        tqdm.write('保存新的BestModel (don\'t stop the run)...')
        self.saveHyperParams()

        # 保存 bestmodel的 的文件名
        model_name = os.path.join(self.modelDir, self.BEST_MODEL_NAME_BASE)
        model_name = model_name + self.MODEL_EXT

        self.best_model = self.best_model_saver.save(sess, model_name)
        tqdm.write('Best Model saved.')


    def loadHyperParams(self):
        """ 读取与当前模型相关的超参数
        """
        # 当前的模型位置(model path)
        self.modelDir = os.path.join(self.args.rootDir, self.MODEL_DIR_BASE)
        if self.args.modelTag:
            print("modelTag=%s" % self.args.modelTag)
            self.modelDir += '-' + self.args.modelTag
        print("modelDir=%s" % self.modelDir)

        # 如果存在config文件，使用其中的一些超参数
        configName = os.path.join(self.modelDir, self.CONFIG_FILENAME)
        if os.path.exists(configName):
            config = configparser.ConfigParser()
            config.read(configName)

            # 恢复超参数
            self.globStep = config['General'].getint('globStep')
            self.args.corpus = config['General'].get('corpus')

            self.args.datasetTag = config['Dataset'].get('datasetTag')
            self.args.maxLength = config['Dataset'].getint('maxLength')
            self.args.filterVocab = config['Dataset'].getint('filterVocab')
            self.args.skipLines = config['Dataset'].getboolean('skipLines')
            self.args.vocabularySize = config['Dataset'].getint('vocabularySize')

            self.args.hiddenSize = config['Network'].getint('hiddenSize')
            self.args.numLayers = config['Network'].getint('numLayers')
            self.args.initEmbeddings = config['Network'].getboolean('initEmbeddings')
            self.args.embeddingSize = config['Network'].getint('embeddingSize')
            self.args.embeddingSource = config['Network'].get('embeddingSource')


    def saveHyperParams(self):
        """ 保存模型的超参数，便于模型管理。
        """
        config = configparser.ConfigParser()
        config['General'] = {}
        config['General']['globStep']  = str(self.globStep)
        config['General']['corpus'] = str(self.args.corpus)

        config['Dataset'] = {}
        config['Dataset']['datasetTag'] = str(self.args.datasetTag)
        config['Dataset']['maxLength'] = str(self.args.maxLength)
        config['Dataset']['filterVocab'] = str(self.args.filterVocab)
        config['Dataset']['skipLines'] = str(self.args.skipLines)
        config['Dataset']['vocabularySize'] = str(self.args.vocabularySize)

        config['Network'] = {}
        config['Network']['hiddenSize'] = str(self.args.hiddenSize)
        config['Network']['numLayers'] = str(self.args.numLayers)
        config['Network']['initEmbeddings'] = str(self.args.initEmbeddings)
        config['Network']['embeddingSize'] = str(self.args.embeddingSize)
        config['Network']['embeddingSource'] = str(self.args.embeddingSource)

        # 保留模型学习使用的超参数，仅仅用于模型管理。
        config['Training (won\'t be restored)'] = {}
        config['Training (won\'t be restored)']['learningRate'] = str(self.args.learningRate)
        config['Training (won\'t be restored)']['batchSize'] = str(self.args.batchSize)
        config['Training (won\'t be restored)']['dropout'] = str(self.args.dropout)

        with open(os.path.join(self.modelDir, self.CONFIG_FILENAME), 'w') as configFile:
            config.write(configFile)


    def getDevice(self):
        """ 根据输入超参数管理设备。
        Return:
            str: 运行程序的设备的名称。
        """
        if self.args.device == 'cpu':
            return '/cpu:0'
        elif self.args.device == 'gpu0':
            return '/gpu:0'
        elif self.args.device == 'gpu1':
            return '/gpu:1'
        elif self.args.device is None:
            return None
        else:
            print('Warning: Error in the device name: {}, use the default device'.format(self.args.device))
            return None
