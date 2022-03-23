# -*- coding: utf-8 -*-
import json

import torch


# list containing all possible tokens (vocabulary)
all_tokens = ['\t', '\n', '    ', '!=', '#COMMENT', '%', '%=', '&', '&=', '(',
 ')', '*', '**', '*=', '+', '+=', ',', '-', '-=', '.', '...', '/', '//', '//=',
 '/=', ':', ';', '<', '<<', '<=', '=', '==', '>', '>=', '>>', '>>=',  '<<=',
 '@', 'ID', 'LIT', '[', ']', '^', '^=', 'and', 'as', 'assert', 'break', 'class',
'continue', 'def', 'del', 'elif', 'else', 'except', 'finally', 'for', 'from',
 'global', 'if', 'import', 'in', 'is', 'lambda', 'not', 'or', 'pass', 'raise',
 'return', 'try', 'while', 'with', 'yield', '{', '|', '|=', '}', '~', 'async',
 'await', '!', '**=', 'nonlocal', 'None', 'EOS', 'FIX_ME']

# index used for converting tokens to one-hot vectors
index = {}
for i,x in enumerate(all_tokens):
    index[x] = i

# dictionary that maps fix_types to their target index
fix_type = {'insert': torch.tensor([0]), 'modify': torch.tensor([1]), 'delete': torch.tensor([2])}


# tokens that when inserted into a given piece of code require no extra whitespace
no_space_tokens = ['\n', '\t', '    ']

def to_tensor(tokens):
    '''
    Function that converts a list (sequence) of tokens to a tensor containing
    the tokens as one-hot vectors.
    
    Unknown tokens (not part of all_tokens) have all entries set to 0.
    
    Shape of resulting tensor: (n, 1, len(all_tokens)) (n=length of sequence).
    '''
    tensor = torch.zeros(len(tokens), 1, len(all_tokens))
    for ti, token in enumerate(tokens):
        try:
            tensor[ti][0][index[token]] = 1
        except KeyError:  # ignore unkown tokens (leave them as zero vector)
            pass
    return tensor



def tokenize(string):
    '''
    Reads a string as input and outputs a sequence of tokens in a list.
    
    Note that this function is only meant to be used to tokenize abstracted 
    code, i.e., code similar to the code provided in the training samples.
    Also this tokenizer does only consider 4 spaces (or tabs) as an indent
    which means fewer spaces will be ommited and more spaces could lead to 
    additional tokens.
    '''
    tokens = []
    delimiters = [':', '(', ')', '.', '[', ']', ',', '\n', '{', '}', '@',
                  '\t', '~', ';']
    token = ''
    idx = 0
    indent = '    '
    string += indent
    while idx < len(string):
        char = string[idx]
        if char == ' ':
            if token != '':
                tokens.append(token)
                token = ''
            if string[idx:idx+4] == indent:
                tokens.append(indent)
                idx += 3
        elif char in delimiters:
            if token != '':
                tokens.append(token)
                token = ''
            if char == '.'and string[idx:idx+3] == '...':
                tokens.append('...')
                idx += 2
            else:
                tokens.append(char)
        elif char in ['<', '>', '=', '!', '+', '-', '*', '/', '^', '%', '&', '|']:
            next_char = string[idx+1]
            if token != '':
                tokens.append(token)
                token = ''
            if next_char == '=':
                tokens.append(char + '=')
                idx += 1
            elif char == '/' and next_char == '/':
                next_next_char = string[idx+2]
                if next_next_char == '=':
                    tokens.append('//=')
                    idx += 2
                else:
                    tokens.append('//')
                    idx += 1
            elif char == '*' and next_char == '*':
                next_next_char = string[idx+2]
                if next_next_char == '=':
                    tokens.append('**=')
                    idx += 2
                else:
                    tokens.append('**')
                    idx += 1
            elif char == '<' and next_char == '<':
                next_next_char = string[idx+2]
                if next_next_char == '=':
                    tokens.append('<<=')
                    idx += 2
                else:
                    tokens.append('<<')
                    idx += 1
            elif char == '>' and next_char == '>':
                next_next_char = string[idx+2]
                if next_next_char == '=':
                    tokens.append('>>=')
                    idx += 2
                else:
                    tokens.append('>>')
                    idx += 1
            else:
                tokens.append(char)
        else:
            token += char
        idx += 1
    return tokens[:-1]


def get_data(path):
    '''
    Function that reads in the sample data and returns training_data and 
    evaluation_data. The first 70 JSON files will always be assigned to
    training_data and the last 30 to evaluation_data.
    
    Note that the sample data needs to have the correct structure.
    The functions tries to convert the chararcter fix_location to a token
    fix_location for easier representation, if this fails the sample is ommited
    from the data.
    
    The returned data consists of the tokenized code (with an EOS token added),
    the correct fix_location, the correct fix_type, and the correct fix_token
    for training or evaluation.
    '''
    training_data = []
    for i in range(70):
        with open(f'{path}/training_{i}.json', 'r') as f:
            data = json.load(f)
        for d in data:
            code = d['wrong_code']
            tokens = tokenize(code)
            tokens.append('EOS') # append EOS token to sequence
            fix_location = d['metadata']['fix_location']
            fix_sequence = tokenize(code[:fix_location])
            for j in range(len(tokens)+1):
                if tokens[:j] == fix_sequence:
                    training_data.append((tokens, j, d['metadata']['fix_type'], d['metadata'].get('fix_token', '')))
                    break
            else:
                # case when fix_location cannot be mapped to token representation
                print(f'training sample {d["metadata"]["id"]}: Could not find fix_location in token representation')
    evaluation_data = []
    for i in range(70, 100):
        with open(f'{path}/training_{i}.json', 'r') as f:
            data = json.load(f)
        for d in data:
            code = d['wrong_code']
            tokens = tokenize(code)
            tokens.append('EOS') # append EOS token to sequence
            fix_location = d['metadata']['fix_location']
            fix_sequence = tokenize(code[:fix_location])
            for j in range(len(tokens)+1):
                if tokens[:j] == fix_sequence:
                    evaluation_data.append((tokens, j, d['metadata']['fix_type'], d['metadata'].get('fix_token', '')))
                    break
            else:
                 # case when fix_location cannot be mapped to token representation
                print(f'training sample {d["metadata"]["id"]}: Could not find fix_location in token representation')
    return training_data, evaluation_data