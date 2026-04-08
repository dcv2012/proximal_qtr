import torch
import torch.nn as nn



class MLP_for_MMR(nn.Module):
    def __init__(self, input_dim, train_params):
        super(MLP_for_MMR, self).__init__()

        self.train_params = train_params
        self.network_width = int(train_params["network_width"])
        self.network_depth = int(train_params["network_depth"])
        self.dropout_prob = train_params.get("dropout_prob", 0.5)

        self.layer_list = nn.ModuleList()
        self.dropout_list = nn.ModuleList()

        for i in range(self.network_depth):
            if i == 0:
                self.layer_list.append(nn.Linear(input_dim, self.network_width))
            else:
                self.layer_list.append(nn.Linear(self.network_width, self.network_width))
            self.dropout_list.append(nn.Dropout(self.dropout_prob))

        self.layer_list.append(nn.Linear(self.network_width, 1))  # 输出层
        self.initialize_weights()  # Initialize the weight

    def initialize_weights(self):
        for layer in self.layer_list:
            if isinstance(layer, nn.Linear):
                torch.nn.init.kaiming_normal_(layer.weight, nonlinearity='relu')
                if layer.bias is not None:
                    torch.nn.init.constant_(layer.bias, 0)

    def forward(self, x):
        for ix, layer in enumerate(self.layer_list[:-1]):
            x = torch.nn.functional.leaky_relu(layer(x))
            x = self.dropout_list[ix](x)  # Dropout
        x = self.layer_list[-1](x)

        # x = torch.nn.functional.softplus(x)
        # x = torch.nn.functional.leaky_relu(negative_slope=0.1)
        return x 
