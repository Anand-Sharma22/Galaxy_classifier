import torch

batchsize = 32
epochs = 50
resize_x = 128
resize_y = 128
input_channels = 3
lr = 0.001
optimizer = torch.optim.Adam
criterion = torch.nn.CrossEntropyLoss()

