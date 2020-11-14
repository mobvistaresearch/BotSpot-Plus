import torch
import torch.nn as nn
import torch.nn.init as init



class MLP(torch.nn.Module):
    def __init__(self, input_size, hidden_size, embeds_desc, num_classes, p):
        super(MLP, self).__init__()

        self.embeds = torch.nn.ModuleList()
        for embed_desc in embeds_desc:
            self.embeds.append(nn.Embedding(embed_desc[0], embed_desc[1]))
            input_size += embed_desc[1]

        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, 32)
        self.fc3 = nn.Linear(32, num_classes)
        self.dropout = nn.Dropout(p)
        self.sigmoid = nn.Sigmoid()



    def forward(self, data):
        x = data[0]
        embed_inputs = data[1]
        inputs = [x]
        for embed_input, embed in zip(embed_inputs, self.embeds):
            embed_output = embed(embed_input)
            inputs.append(embed_output)

        x = torch.cat(inputs, 1)
        x = self.fc1(x)
        x = self.dropout(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.dropout(x)
        x = self.relu(x)
        x = self.fc3(x)
        x = self.sigmoid(x)
        return x
