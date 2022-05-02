import torch.nn as nn
import torch

class AutoEncoder(nn.Module):
    def __init__(self, activation=nn.ReLU(),
                 input_size=3*8*8, hidden_sizes=[32*3,8,32*3]):
        super(AutoEncoder, self).__init__()
        self.input_size = input_size
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.to(device)        
        self.activation = activation
        self.fc1 = nn.Linear(input_size, hidden_sizes[0])
        self.fc2 = nn.Linear(hidden_sizes[0], hidden_sizes[1])
        self.fc3 = nn.Linear(hidden_sizes[1], hidden_sizes[2])
        self.fc4 = nn.Linear(hidden_sizes[2], input_size)


    def forward(self, x):
        x = self.encode(x)
        x = self.decode(x)

        return x
    
    def encode(self,x):
        x = x.view(-1, self.input_size)
        x = self.fc1(x)
        x = self.activation(x)
        x = self.fc2(x)

        return x
    
    def decode(self,x):        
        x = self.activation(x)
        x = self.fc3(x)        
        x = self.activation(x)
        x = self.fc4(x)
        sig = nn.Sigmoid()
        x = sig(x)

        return x
    
    def save_model(self,PATH):
        torch.save(self.state_dict(),PATH)
    
    def load_autoencoder(PATH):
        model = AutoEncoder()
        model.load_state_dict(torch.load(PATH))
        model.eval()
        return model
