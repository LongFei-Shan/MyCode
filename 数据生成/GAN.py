from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.losses import BinaryCrossentropy
import tensorflow.keras as keras
import tensorflow as tf
import numpy as np
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"


class GAN_FCNN:
    def __init__(self, lr, noiseDim, sampleDim, batchSize, epoch, generatorShape=(3, 5), discriminatorShape=(5, 3), activation="tanh",
                 save_path="./GAN", output_g_activition=None, output_d_activition="sigmoid", up_limit=1, down_limit=-1, training=True,
                 seed=0, is_ouput_train_result=True):
        """
        :param lr: 学习率
        :param noiseDim: 噪声数据维度
        :param sampleDim: 样本数据维度
        :param batchSize: 每批次训练数据个数
        :param epoch: 训练多少批次
        :param generatorShape: 生成器中间层神经元个数，一共两个隐藏层，需要输入两个隐藏层的神经元个数
        :param discriminatorShape: 判别器中间层神经元个数，一共两个隐藏层，需要输入两个隐藏层的神经元个数
        :param activation: 激活函数
        :param save_path: 模型存储路径
        :param output_g_activition: 生成器输出层激活函数
        :param output_d_activition: 判别器输出层激活函数
        :param up_limit: 生成噪声的上界，采用np.random.uniform()进行生成
        :param down_limit: 生成噪声的下界，采用np.random.uniform()进行生成
        :param training: 是否进行训练
        :param seed: 随机数种子，只有在使用cpu时才有效
        :param is_ouput_train_result: 是否输出训练过程loss结果等
        """
        # 初始化参数
        # 是否使用GUP
        self.lr = lr
        self.noiseDim = noiseDim
        self.sampleDim = sampleDim
        self.generatorShape = generatorShape
        self.epoch = epoch
        self.discriminatorShape = discriminatorShape
        self.activation = activation
        self.batchSize = batchSize
        self.save_path = save_path
        self.output_g_activition = output_g_activition
        self.output_d_activition = output_d_activition
        self.up_limit = up_limit
        self.down_limit = down_limit
        self.training = training
        self.is_ouput_train_result = is_ouput_train_result
        # 损失函数
        self.lossFunc = BinaryCrossentropy(from_logits=False)
        # 优化器
        self.generator_opt = keras.optimizers.Adam(self.lr)
        self.discriminator_opt = keras.optimizers.Adam(self.lr)
        # 初始化网络
        self.model_d = self.discriminator_model()
        self.model_g = self.generator_model()
        # 随机树种子
        tf.random.set_seed = seed

    def generator_model(self):
        """生成器"""
        model1 = Sequential()
        model1.add(Dense(self.generatorShape[0], activation=self.activation))
        model1.add(Dense(self.generatorShape[1], activation=self.activation))
        model1.add(Dense(self.sampleDim, activation=self.output_g_activition))

        return model1

    def discriminator_model(self):
        """判别器"""
        model2 = Sequential()
        model2.add(Dense(self.discriminatorShape[0], activation=self.activation))
        model2.add(Dense(self.discriminatorShape[1], activation=self.activation))
        model2.add(Dense(1, activation=self.output_d_activition))

        return model2

    def generator_loss(self, fake_out):
        loss = self.lossFunc(tf.ones_like(fake_out), fake_out)
        return loss

    def discriminator_loss(self, fake_out, real_out):
        loss = self.lossFunc(tf.ones_like(real_out), real_out) + self.lossFunc(tf.zeros_like(fake_out), fake_out)
        return loss

    def train_step(self, X, noise):
        # 训练
        with tf.GradientTape() as gen_tape:
            self.gen_data = self.model_g(noise, training=self.training)
            self.fake_out = self.model_d(self.gen_data, training=self.training)
            self.gen_loss = self.generator_loss(self.fake_out)
        self.gradient_gen = gen_tape.gradient(self.gen_loss, self.model_g.trainable_variables)
        self.generator_opt.apply_gradients(zip(self.gradient_gen, self.model_g.trainable_variables))
        with tf.GradientTape() as disc_tape:
            self.real_out = self.model_d(X, training=self.training)
            self.gen_data = self.model_g(noise, training=self.training)
            self.fake_out = self.model_d(self.gen_data, training=self.training)
            self.disc_loss = self.discriminator_loss(self.fake_out, self.real_out)
        self.gradient_disc = disc_tape.gradient(self.disc_loss, self.model_d.trainable_variables)
        self.discriminator_opt.apply_gradients(zip(self.gradient_disc, self.model_d.trainable_variables))

    def fit(self, X):
        """
        :param X: 训练样本数据
        :return:
        """
        for loop in range(self.epoch):
            # 生成噪声
            noise = np.random.uniform(self.down_limit, self.up_limit, [self.batchSize, self.noiseDim]).astype(np.float32)
            for batch in range(len(X)//self.batchSize):
                subData = X[batch*self.batchSize:(batch+1)*self.batchSize, :]
                self.train_step(subData, noise)
            if self.is_ouput_train_result:
                print(f"epoch={loop}, gen_loss={self.gen_loss.numpy()}, disc_loss={self.disc_loss.numpy()}, "
                      f"fake_out={np.mean(self.fake_out)}, real_out={np.mean(self.real_out)}")
        keras.Model.save(self.model_g, f"{self.save_path}/generator/")
        keras.Model.save(self.model_d, f"{self.save_path}/discriminator/")

    def predict(self, length):
        """
        :param length: 生成数据个数
        :return:
        """
        model_g = keras.models.load_model(f"{self.save_path}/generator/")
        # 生成噪声
        noise = np.random.uniform(self.down_limit, self.up_limit, [length, self.noiseDim]).astype(np.float32)
        # 生成数据
        gen_data = model_g(noise, training=False)

        return gen_data.numpy()

