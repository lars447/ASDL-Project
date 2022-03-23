#!/usr/bin/python3
import argparse
import random
import datetime

import torch
import torch.nn as nn

import Utils

parser = argparse.ArgumentParser()
parser.add_argument(
    '--source', help="Folder path of all training files.", required=True)
parser.add_argument(
    '--destination', help="Path to save your trained model.", required=True)

class Location_Model(nn.Module):
    
    def __init__(self, input_size, hidden_size):
        super().__init__()
        
        self.hidden_size = hidden_size
        self.bi_lstm = nn.LSTM(input_size, hidden_size, bidirectional=True)
        self.h2o = nn.Linear(hidden_size*2, 1)

        self.loss_fn = nn.CrossEntropyLoss()
    
    def forward(self, inp):
        output, hidden = self.bi_lstm(inp)
        return torch.reshape(self.h2o(output), (1, inp.shape[0]))


class Type_Model(nn.Module):
    
    def __init__(self, input_size, hidden_size):
        super().__init__()
        
        self.hidden_size = hidden_size
        self.bi_lstm = nn.LSTM(input_size, hidden_size, bidirectional=True)
        self.h2o = nn.Linear(hidden_size*2, 3)
        self.dropout = nn.Dropout(p=0.35, inplace=False)
        
        self.loss_fn = nn.CrossEntropyLoss()
    
    def forward(self, inp):
        context, (out, _) = self.bi_lstm(inp)
        hidden = self.dropout(torch.reshape(out, (1, 2*self.hidden_size)))
        output = self.h2o(hidden)
        return output
    
    
class Token_Model(nn.Module):
    
    def __init__(self, input_size, hidden_size):
        super().__init__()
        
        self.hidden_size = hidden_size
        self.bi_lstm = nn.LSTM(input_size, hidden_size, bidirectional=True)
        self.h2o = nn.Linear(hidden_size*2, input_size)
        self.dropout = nn.Dropout(p=0.5, inplace=False)
        
        self.loss_fn = nn.CrossEntropyLoss()
    
    def forward(self, inp):
        context, (out, _) = self.bi_lstm(inp)
        hidden = self.dropout(torch.reshape(out, (1, 2*self.hidden_size)))
        output = self.h2o(hidden)
        return output


def train_location_model(model, source):
    training_data, evaluation_data = Utils.get_data(source)
    
    # training loop
    model.train()
    print('Training Location Model...')
    for epoch in range(n_epochs):
        random.shuffle(training_data)
        c_loss = 0
        for data in training_data:
            tokens = data[0][:]
            inp_tensor = Utils.to_tensor(tokens)
            inp_tensor = inp_tensor.to(device)
            target_tensor = torch.tensor([data[1]])
            target_tensor = target_tensor.to(device)
            
            model.zero_grad()

            output = model(inp_tensor)

            loss = model.loss_fn(output, target_tensor)
            loss.backward()

            # Add parameters' gradients to their values, multiplied by learning rate
            for p in model.parameters():
                p.data.add_(p.grad.data, alpha=-lr_location)
            c_loss += loss.item()
        print(epoch+1, c_loss)
        
    # evaluation
    model.eval()
    print('Evaluating Location Model...')
    correct = 0
    with torch.no_grad():
        for data in evaluation_data:
            inp_tensor = Utils.to_tensor(data[0])
            inp_tensor = inp_tensor.to(device)
            target_tensor = torch.tensor(data[1])
            target_tensor = target_tensor.to(device)
            output = model(inp_tensor)
            prediction = torch.argmax(output)
            if prediction == target_tensor:
                correct += 1
    print('Accuracy:', correct/len(evaluation_data))
        
        
def train_type_model(model, source):
    optimizer = torch.optim.Adam(model.parameters(), lr=lr_type)
    training_data, evaluation_data = Utils.get_data(source)
    
    #training loop
    model.train()
    print('Training Type Model...')
    for epoch in range(n_epochs):
        random.shuffle(training_data)
        c_loss = 0
        for data in training_data:
            tokens = data[0][:]
            tokens.insert(data[1], 'FIX_ME')  # insert FIX_ME at fix_location
            inp_tensor = Utils.to_tensor(tokens)
            inp_tensor = inp_tensor.to(device)
            target_tensor = Utils.fix_type[data[2]]  # create tensor based on fix type
            target_tensor = target_tensor.to(device)
            
            optimizer.zero_grad()
            
            output = model(inp_tensor)

            loss = model.loss_fn(output, target_tensor)
            loss.backward()
            
            # update parameters
            optimizer.step()
            
            c_loss += loss.item()
        print(epoch+1, c_loss)

    # evaluation
    model.eval()
    print('Evaluating Type Model...')
    correct = 0
    with torch.no_grad():
        for data in evaluation_data:
            tokens = data[0][:]
            tokens.insert(data[1], 'FIX_ME')  # insert FIX_ME token
            inp_tensor = Utils.to_tensor(tokens)
            inp_tensor = inp_tensor.to(device)
            target_tensor = Utils.fix_type[data[2]]  # create tensor based on fix type
            target_tensor = target_tensor.to(device)
            output = model(inp_tensor)
            prediction = torch.argmax(output)
            if prediction == target_tensor:
                correct += 1
    print('Accuracy:', correct/len(evaluation_data))
    

def train_token_model(model, source):
    optimizer = torch.optim.Adam(model.parameters(), lr=lr_type)
    training_data, evaluation_data = Utils.get_data(source)
    
    #training loop
    model.train()
    print('Training Token Model...')
    for epoch in range(n_epochs):
        random.shuffle(training_data)
        c_loss = 0
        for data in training_data:
            tokens = data[0][:]
            # adjust tokens based on fix_type, skip deletions
            if data[2] == 'insert':
                tokens.insert(data[1], 'FIX_ME')  # insert FIX_ME at fix_location
            elif data[2] == 'modify':
                tokens[data[1]] = 'FIX_ME'  # change wrong token to FIX_ME
            else:
                continue
            inp_tensor = Utils.to_tensor(tokens)
            inp_tensor = inp_tensor.to(device)
            try:
                target_tensor = torch.tensor([Utils.index[data[3]]])
            except KeyError:
                continue  # skip samples where fix_token is not known
            target_tensor = target_tensor.to(device)
            
            optimizer.zero_grad()
            
            output = model(inp_tensor)

            loss = model.loss_fn(output, target_tensor)
            loss.backward()
            
            # update parameters
            optimizer.step()
            
            c_loss += loss.item()
        print(epoch+1, c_loss)
        
    # evaluation
    model.eval()
    print('Evaluating Token Model...')
    correct = 0
    num_skipped = 0
    with torch.no_grad():
        for data in evaluation_data:
            tokens = data[0][:]
            # adjust tokens based on fix_type, skip deletions
            if data[2] == 'insert':
                tokens.insert(data[1], 'FIX_ME')  # insert FIX_ME at fix_location
            elif data[2] == 'modify':
                tokens[data[1]] = 'FIX_ME'  # change wrong token to FIX_ME
            else:
                num_skipped += 1
                continue
            inp_tensor = Utils.to_tensor(tokens)
            inp_tensor = inp_tensor.to(device)
            try:
                target_tensor = torch.tensor([Utils.index[data[3]]])
            except KeyError:
                num_skipped += 1
                continue  # skip samples where fix_token is not known
            target_tensor = target_tensor.to(device)
            output = model(inp_tensor)
            prediction = torch.argmax(output)
            if prediction == target_tensor:
                correct += 1
    print('Accuracy:', correct/(len(evaluation_data) - num_skipped))


def save_model(model, destination):
    torch.save(model, destination)
	

if __name__ == "__main__":
    args = parser.parse_args()
    # global constants 
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    inp_size = len(Utils.all_tokens)
    hidden_size = 128
    lr_location = 0.005
    lr_type = 0.001
    lr_token = 0.001
    n_epochs = 20
    start = datetime.datetime.now()
    location_model = Location_Model(inp_size, hidden_size)
    location_model.to(device)
    
    # train and save model for predicting fix_location
    train_location_model(location_model, args.source)
    save_model(location_model, args.destination + '/location_model.pt')
    print('Time elapsed:', datetime.datetime.now() - start)
    
    type_model = Type_Model(inp_size, hidden_size)
    type_model.to(device)
    
    # train and save model for predicting fix_type
    train_type_model(type_model, args.source)
    save_model(type_model, args.destination + '/type_model.pt')
    print('Time elapsed:', datetime.datetime.now() - start)
    
    token_model = Token_Model(inp_size, hidden_size)
    token_model.to(device)
    
    # train and save mode for predicting token
    train_token_model(token_model, args.source)
    save_model(token_model, args.destination + '/token_model.pt')
    print('Time elapsed:', datetime.datetime.now() - start)
