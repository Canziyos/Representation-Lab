# import torch
# from torch.utils.data import DataLoader

# class Trainer:
#     """
#     Simple training loop interface. 
#     """
#     def __init__(self, model, optimizer, criterion, device, teacher_forcing_ratio=0.5):
#         self.model = model
#         self.optimizer = optimizer
#         self.criterion = criterion
#         self.device = device
#         self.teacher_forcing_ratio = teacher_forcing_ratio

#     def train_one_epoch(self, dataloader):
#         self.model.train()
#         total_loss = 0.0

#         for src, trg in dataloader:
#             # src, trg come in as lists or Tensors
#             src = src.to(self.device)
#             trg = trg.to(self.device)

#             self.optimizer.zero_grad()
#             output = self.model(src, trg, self.teacher_forcing_ratio)
#             # output: (batch_size, trg_len, output_dim)

#             # Flatten to match CrossEntropyLoss shape
#             output = output.view(-1, output.shape[-1])
#             trg = trg.view(-1)

#             loss = self.criterion(output, trg)
#             loss.backward()
#             self.optimizer.step()
#             total_loss += loss.item()

#         return total_loss / len(dataloader)

#     def evaluate(self, dataloader):
#         self.model.eval()
#         total_loss = 0.0

#         with torch.no_grad():
#             for src, trg in dataloader:
#                 src = src.to(self.device)
#                 trg = trg.to(self.device)

#                 output = self.model(src, trg, teacher_forcing_ratio=0.0)
#                 output = output.view(-1, output.shape[-1])
#                 trg = trg.view(-1)

#                 loss = self.criterion(output, trg)
#                 total_loss += loss.item()

#         return total_loss / len(dataloader)
