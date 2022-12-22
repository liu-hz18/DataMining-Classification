import torch, random
import torch.nn.functional as F
import numpy as np

class MLPModel(torch.nn.Module):
    def __init__(self, input_dim, h1, h2, output_dim):
        super().__init__()
        print('MLP shape:', input_dim, h1, h2, output_dim)
        self.fc1 = torch.nn.Linear(input_dim, h1)  
        self.fc2 = torch.nn.Linear(h1, h2)
        self.fc3 = torch.nn.Linear(h2, output_dim)
        self.dropout = torch.nn.Dropout(p=0.5)

    def forward(self, inputs):
        x = self.dropout(F.relu(self.fc1(inputs)))
        x = self.dropout(F.relu(self.fc2(x)))
        logits = self.fc3(x)
        return logits

class MultiLayerPerceptronClassifier:
    def __init__(self, random_state, lr=1e-2, h1=512, h2=128, epoch=8, batch_size=64):
        super().__init__()
        torch.manual_seed(random_state)
        torch.cuda.manual_seed_all(random_state)
        np.random.seed(random_state)
        random.seed(random_state)
        self.lr = lr
        self.h1 = h1
        self.h2 = h2
        self.epoch = epoch
        self.batch_size = batch_size
        self.loss_func = torch.nn.CrossEntropyLoss()

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

    def fit(self, data, label):
        dataloader = self.DataProcess(data, label=label, shuffle=True)
        self.model = MLPModel(self.input_dim, self.h1, self.h2, 2).cuda()
        self.optimizer = torch.optim.Adam(params=self.model.parameters(), lr=self.lr)

        self.model.train()
        for e in range(self.epoch):
            tr_loss, tr_steps = 0, 0
            for step, batch in enumerate(dataloader):
                batch = tuple(t.to('cuda') for t in batch)
                b_inputs, b_labels = batch

                self.optimizer.zero_grad()
                b_logits = self.model(b_inputs)
                b_loss = self.loss_func(b_logits, b_labels)

                b_loss.backward()
                self.optimizer.step()

                tr_loss += b_loss.item()
                tr_steps += 1

    def predict(self, data):
        self.model.eval()
        dataloader = self.DataProcess(data, label=None, shuffle=False)
        logits = []
        for step, batch in enumerate(dataloader):
            batch = tuple(t.to('cuda') for t in batch)
            b_inputs = batch[0]
            with torch.no_grad():
                b_logits = self.model(b_inputs)
            logits.append(b_logits.cpu())
        logits = torch.cat([_ for _ in logits], dim=0)
        preds = torch.argmax(logits, -1)
        return preds