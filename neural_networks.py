import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

class NN(nn.Module):
    def __init__(self, device, lr, layers_size):
        super(NN, self).__init__()
        
        self.number_of_layers = len(layers_size)
        
        for i in range(self.number_of_layers - 1):
            setattr(self, f'fc{i + 1}', nn.Linear(layers_size[i], layers_size[i + 1]))
            
        self.optimizer = optim.Adam(self.parameters(), lr=lr)
        self.loss = nn.MSELoss()
        self.to(device)
        
    def forward(self, input_data):
        
        layer1 = F.relu(self.fc1(input_data))
        
        # Ugly code used to avoid 
        # RuntimeError: Only Tensors created explicitly by the user (graph leaves) support the deepcopy protocol at the moment
        # when using deepcopy
        
        if self.number_of_layers < 4:
            return self.fc2(layer1)
        else:
            layer2 = F.relu(self.fc2(layer1))
            if self.number_of_layers < 5:
                return self.fc3(layer2)
            else:
                layer3 = F.relu(self.fc3(layer2))
                if self.number_of_layers < 6:
                    return self.fc4(layer3)
                else:
                    layer4 = F.relu(self.fc4(layer3))
                    if self.number_of_layers < 7:
                        return self.fc5(layer4)
                    else:
                        layer5 = F.relu(self.fc5(layer4))
                        if self.number_of_layers < 8:
                            return self.fc6(layer5)
                        else:
                            layer6 = F.relu(self.fc6(layer5))
                            if self.number_of_layers < 9:
                                return self.fc7(layer6)
                            else:
                                raise('More than 8 layers not implemented')
        
        # Won't work:
        
        # for i in range(1, self.number_of_layers - 2):
        #     setattr(self, f'layer{i + 1}', F.relu(getattr(self, f'fc{i + 1}')(getattr(self, f'layer{i}'))))
            
        # return getattr(self, f'fc{self.number_of_layers - 1}')(getattr(self, f'layer{self.number_of_layers - 2}'))

        # Won't work too:
            
        # self.layers = [F.relu(self.fc0(input_data))]
        
        # for i in range(1, self.number_of_layers - 2):
        #     self.layers.append(F.relu(getattr(self, f'fc{i}')(self.layers[-1])))
        
        # return getattr(self, f'fc{self.number_of_layers - 2}')(self.layers[-1])








