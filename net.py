# -*-codeing=utf-8-*-
# @Time：2021/12/8 21:14
# @Autor:李薇
# File: 1.PY
# @Software: PyCharm
# encoding=utf-8
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import time
import random

from torch.nn import Linear

from layers import GraphConvolution


class RelationWork(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_class):
        super(RelationWork, self).__init__()
        self.hidden_dim = hidden_dim
        self.input_dim = input_dim
        self.num_class = num_class
        self.weight = nn.Parameter(torch.zeros(size=[input_dim, hidden_dim]))
        nn.init.xavier_uniform_(self.weight.data, gain=1.414)
        self.bias = nn.Parameter(torch.zeros(hidden_dim))
        nn.init.uniform_(self.bias.data, a=0, b=1)

        # 下面，定义一个单层的神经网络来实现Z函数
        self.W1 = nn.Parameter(torch.zeros(size=[input_dim*2, hidden_dim]))
        nn.init.xavier_uniform_(self.W1.data, gain=1.414)
        self.Bias1 = nn.Parameter(torch.zeros(hidden_dim))
        nn.init.uniform_(self.Bias1.data, a=0, b=1)
        self.W2 = nn.Parameter(torch.zeros(size=[hidden_dim , num_class]))
        nn.init.xavier_uniform_(self.W2.data, gain=1.414)
        self.Bias2 = nn.Parameter(torch.zeros(num_class))
        nn.init.uniform_(self.Bias2.data, a=0, b=1)

        self.lin1 = Linear(input_dim*2, hidden_dim)
        self.lin2 = Linear(hidden_dim , num_class)


    def f(self, x):
        '''
        定义编码函数
        :param x: 需要编码的数据
        :return:  编码之后的结果
        '''
        embeddings = torch.matmul(x, self.weight) + self.bias
        embeddings = F.relu(embeddings)
        return embeddings

    def Z(self, S,Q):
        '''
        :param x_i: Support Set中的编码数据
        :param x_j: Query Set中编码数据
        :return:通过计算得到的相关性结果
        # 100 x 16 query
        # 5 x 16 prototype
        '''
        # rows = prototype_embeddings.shape[0]
        # x_j = query_embeddings.repeat((rows, 1))
        n = Q.size(0)
        m = S.size(0)
        d = S.size(1)
        assert d == S.size(1)

        Q = Q.unsqueeze(1).expand(n, m, d)
        S = S.unsqueeze(0).expand(n, m, d)
        con = torch.cat([S, Q], dim=2)
        # print(x.shape,self.W.shape)
        # x:torch.Size([100,5, 32]),x.sum(1)是[5,32]
        # W:torch.Size([32, 5])
        #两个线性层
        # z1 = torch.matmul(x.sum(1), self.W1) + self.Bias1
        # a1=torch.relu(z1)
        # z2=torch.matmul(a1, self.W2) + self.Bias2
        # result = F.relu(z2)
        con=con.sum(1)
        z1 = self.lin1(con)
        z2 = z1.relu()
        z2 = F.dropout(z2, p=0.5, training=self.training)
        result = self.lin2(z2)
        return result



    def g(self, x):
        '''
        定义g函数，这里使用softmax函数 转换成score
        :param x: 通过Z计算出来的相关性结果
        :return: 当前样本属于各个分类的概率
        '''
        return F.softmax(x, dim=1)

    def forward(self,prototype_embeddings,query_embeddings):
        '''
        基本过程：
        1. 编码
        2. 相似性计算
        3. 转换成分值
        :param x_i: 支持集中的原始数据
        :param x_j: 查询集中的原始数据
        :return:
        '''
        #x_i变为支持集的嵌入support_embeddings
        #x_j变为查询集的嵌入
        #classall理解为prototype_embeddings
        # fxi = self.f(prototype_embeddings)
        # fxj = self.f(query_embeddings)
        similar = self.Z(prototype_embeddings,query_embeddings)
        return self.g(similar)





