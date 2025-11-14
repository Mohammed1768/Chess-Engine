import sys
import os
import torch
import torch.nn as nn
import chess
import numpy as np

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from Engine.ChessNet import ChessNet
from Engine.Encoder import Encoder



net = ChessNet()
net.load_state_dict(torch.load("chessnet_weights.pth", map_location="cpu"))
net.eval()


class Board(chess.Board):
    def __init__(self):
        super().__init__()


    def get_matrix_view(self):
        encoder = Encoder()
        return encoder.encode_FEN(self.fen())


    def get_eval(self):
        planes = self.get_matrix_view()
        input_tensor = torch.tensor(planes, dtype=torch.float32).unsqueeze(0) 

        with torch.no_grad():
            output = net(input_tensor)

        return output
