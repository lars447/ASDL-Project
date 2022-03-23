#!/usr/bin/python3

import argparse
import json
from pathlib import Path

import torch
import torch.nn as nn

import Utils

parser = argparse.ArgumentParser()
parser.add_argument(
    '--model', help="Path of trained models.", required=True)
parser.add_argument(
    '--source', help="Folder path of all test files.", required=True)
parser.add_argument(
    '--destination', help="Path to output json file of extracted predictions.", required=True)

# class for each model 
###############################################################################
class Location_Model(nn.Module):
    
    def __init__(self, input_size, hidden_size):
        super().__init__()
        
        self.hidden_size = hidden_size
        self.bi_lstm = nn.LSTM(input_size, hidden_size, bidirectional=True)
        self.h2o = nn.Linear(hidden_size*2, 1)

        self.loss_fn = nn.CrossEntropyLoss()
    
    def forward(self, inp):
        output, hidden = self.bi_lstm(inp)
        return torch.reshape(self.h2o(output), (1,inp.shape[0]))


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
###############################################################################


def from_token_to_char(code, token):
    '''
    Function that converts the fix_location from a token level to a character
    level.
    '''
    # add special EOS token to code, make sure an accidental indent isn't added
    if code.endswith('   '):
        code += 'EOS'
    else:
        code += ' EOS'
    token_index = token.item()
    tokens = Utils.tokenize(code)
    token = tokens[token_index]
    n_token = 0
    for i in range(token_index+1):
        if tokens[i] == token:
            n_token += 1
    position = -1
    for i in range(n_token):
        position = code.find(token, position+1)
    if token == 'EOS':  # adjust position for EOS token
        position -= 1
    return position


def predict(models, test_files):
    predictions = []
    for path in Path(test_files).glob('*.json'):
        with path.open() as f:
            data = json.load(f)
        for d in data:
            # find fix_location
            code = d['wrong_code']
            tokens = Utils.tokenize(code)
            tokens.append('EOS') # append EOS token to sequence
            tensor = Utils.to_tensor(tokens)
            tensor = tensor.to(device)
            output = models[0](tensor)
            location = torch.argmax(output)  # prediction on token level
            fix_location = from_token_to_char(code, location)
            
            # find fix_type
            wrong_token = tokens[location.item()]  # token that needs to be modified/removed if fix_type is not insert
            tokens.insert(location.item(), 'FIX_ME')
            tensor = Utils.to_tensor(tokens)
            tensor = tensor.to(device)
            output = models[1](tensor)
            fix_type = torch.argmax(output)
            if fix_type.item() != 2:  # fix_type not delete
                # adjust tokens based on fix_type
                tokens = Utils.tokenize(code)
                tokens.append('EOS')
                if fix_type.item() == 0:
                    fix_type = 'insert'
                    tokens.insert(location.item(), 'FIX_ME')  # insert FIX_ME at fix_location
                elif fix_type.item() == 1:
                    fix_type = 'modify'
                    tokens[location.item()] = 'FIX_ME'  # change wrong token to FIX_ME
                else:
                    continue
                
                tensor = Utils.to_tensor(tokens)
                tensor = tensor.to(device)
                output = models[2](tensor)
                token = torch.argmax(output)
                token = Utils.all_tokens[token.item()]
            else:
                fix_type = 'delete'
                token = ''
            metadata = {'file': d['metadata']['file'], 'id': d['metadata']['id']}
            
            if fix_type == 'insert':
                if token in Utils.no_space_tokens:
                    fix_token = token
                else:
                    fix_token = token + ' '
                fixed_code = code[:fix_location] + fix_token + code[fix_location:]
            elif fix_type == 'modify':
                if wrong_token + ' ' in code[fix_location:] and token == '\n' and tokens[location.item() + 1] != '    ':
                    wrong_token = wrong_token + ' '
                fixed_code = code[:fix_location] + code[fix_location:].replace(wrong_token, token, 1)
            else:  # fix_type == delete
                if wrong_token + ' ' in code[fix_location:] and tokens[location.item() + 1] != '    ':
                    wrong_token = wrong_token + ' '
                fixed_code = code[:fix_location] + code[fix_location:].replace(wrong_token, '', 1)
            
            if token:
                predictions.append({'metadata': metadata,
                                    'predicted_code': fixed_code,
                                    'predicted_location': fix_location,
                                    'predicted_type': fix_type,
                                    'predicted_token': token})
            else:
                predictions.append({'metadata': metadata,
                                    'predicted_code': fixed_code,
                                    'predicted_location': fix_location,
                                    'predicted_type': fix_type})
    return sorted(predictions, key=lambda x: x['metadata']['id'])

def load_model(source):
    """
    Load all models needed for fixing the syntax error. This includes the model
    for localizing the error, the model for predicting the fix type, and the
    model for predicting the fix token.
    """
    models = []
    location_model = torch.load(source + '/location_model.pt', map_location=device)
    location_model.to(device)
    location_model.eval()
    models.append(location_model)
    type_model = torch.load(source + '/type_model.pt', map_location=device)
    type_model.to(device)
    type_model.eval()
    models.append(type_model)
    token_model = torch.load(source + '/token_model.pt', map_location=device)
    token_model.to(device)
    token_model.eval()
    models.append(token_model)
    return models

def write_predictions(destination,predictions):
    s = json.dumps(predictions, indent=2)
    with open(destination, 'w') as f:
        f.write(s)

if __name__ == "__main__":
    args = parser.parse_args()
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    
    # load the serialized model
    models = load_model(args.model)

    # predict incorrect location for each test example.
    predictions = predict(models, args.source)

    # write predictions to file
    write_predictions(args.destination, predictions)
