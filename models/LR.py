import torch
import numpy as np
import random
import pandas as pd
from sklearn.linear_model import LogisticRegression

class myLRModel:
    def __init__(self, batch_size=64, lr=1e-2, epoch=3, threshhold=0.5, random_state=42):
        self.batch_size = batch_size
        self.lr = lr
        self.epoch = epoch
        self.threshhold = threshhold
        torch.manual_seed(random_state)
        torch.cuda.manual_seed_all(random_state)
        np.random.seed(random_state)
        random.seed(random_state)

    def sigmoid(self, features):
        return 1 / (1 + np.exp(-np.dot(features, self.w)))

    def DataProcess(self, data, label=None, shuffle=False):
        inputs = torch.tensor(data).float()
        self.input_dim = inputs.shape[1]
        if label is not None:
            labels = torch.tensor([int(_) for _ in label]).long()
            dataset = torch.utils.data.TensorDataset(inputs, labels)
        else:
            dataset = torch.utils.data.TensorDataset(inputs)
        if shuffle:
            sampler = torch.utils.data.RandomSampler(dataset)
        else:
            sampler = torch.utils.data.SequentialSampler(dataset)
        dataloader = torch.utils.data.DataLoader(dataset, sampler=sampler, batch_size=self.batch_size)
        return dataloader

    def fit(self, data, labels):
        dataloader = self.DataProcess(data, labels)
        self.w = np.random.normal(0, 0.01, size=(self.input_dim, 1))
        self.b = np.random.normal(0, 0.01)
        for e in range(self.epoch):
            for step, batch in enumerate(dataloader):
                b_inputs, b_labels = batch
                b_inputs, b_labels = b_inputs.numpy(), b_labels.numpy()
                self.w -= self.lr * np.dot(np.mat(b_inputs).T, self.sigmoid(b_inputs) - b_labels.reshape(-1, 1))

    def predict(self, data):
        return self.sigmoid(data) >= self.threshhold
