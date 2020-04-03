import torch
import torch.nn as nn
import torchvision.models as models


class EncoderCNN(nn.Module):
    def __init__(self, embed_size):
        super(EncoderCNN, self).__init__()
        resnet = models.resnet50(pretrained=True)
        for param in resnet.parameters():
            param.requires_grad_(False)
        
        modules = list(resnet.children())[:-1]
        print(modules)
        self.resnet = nn.Sequential(*modules)
        self.embed = nn.Linear(resnet.fc.in_features, embed_size)

    def forward(self, images):
        features = self.resnet(images)
        features = features.view(features.size(0), -1)
        features = self.embed(features)
        return features

class DecoderRNN(nn.Module):
    def __init__(self, embed_size, hidden_size, vocab_size, num_layers=1):
        super().__init__()
        #defining word embedding layer
        
        self.num_layers = num_layers
        self.hidden_size = hidden_size
     
        
        self.word_embed = nn.Embedding(vocab_size, embed_size)
        
        #defining lstm cell
        self.lstm = nn.LSTM(input_size=embed_size, hidden_size=hidden_size, num_layers=num_layers, batch_first=True)
        #self.hidden = self.init_hidden()
        
        #defining final linear layer, that produces outputs
        self.linear = nn.Linear(hidden_size, vocab_size)
    
    def forward(self, features, captions):
        #removing <end> token from captions
        captions = captions[:, :-1]
        #passing captions through embedding layer
        captions = self.word_embed(captions)
        print("Shape of captions")
        print(captions.shape)
        #concantenating features (from CNN output) with embedded captions
        inputs = torch.cat((features.unsqueeze(1), captions), dim=1)
        print("Shape of inputs")
        print(inputs.shape)
        
        #forward pass through the LSTM cell
        ht, _ = self.lstm(inputs)
        
        #passing Ht through linear layer to obtain final outputs
        y = self.linear(ht)
        
        return y

    def sample(self, inputs, states=None, max_len=20):
        " accepts pre-processed image tensor (inputs) and returns predicted sentence (list of tensor ids of length max_len) "
        
        #remember that the unfolded rnn model is simply an abstraction
        #in actuality, we are dealing with just 1 cell!
        #this why I´m using a loop
        #for the unfolding of the cell through time to be coherent, hidden (tuple with cell and hidden state) must flow from one cell abstraction to the other
        
        outputs = []
        output_len = 0
        
        #initializing hidden states and moving them to the current device
        states = (torch.randn(1, 1, self.hidden_size).to(inputs.device), 
                  torch.randn(1, 1, self.hidden_size).to(inputs.device))
        
        while output_len < max_len:
            
            
            #forward pass through the decoder
            ht, states = self.lstm(inputs, states)
            #collasing batch_size dim before passing data onto linear output layer
            y = self.linear(ht.squeeze(dim=1))
            
            #obtain predicted index. torch.max returns (values, indices) tuple. 
            predicted_value, predicted_index = torch.max(y, 1)
            
            #no need to continue producing outputs if we hit an end token
            if predicted_index == 1:
                break
            
            #tapping in the relevant content of the tensor
            #appending prediction to outputs container
            outputs.append(predicted_index.cpu().numpy()[0].item())
            
            #output of LSTM at t=1 serves as input for LSTM at t=t+1
            #must go through embedding layer first, 'un-collapsing' batch_size dimension
            inputs = self.word_embed(predicted_index).unsqueeze(1)
            
           
            print(inputs.shape)
            print(inputs)
        
            output_len += 1
        
        
        return outputs
        