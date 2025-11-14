import sys
import os
import torch
import torch.nn as nn
import chess

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ChessNet import ChessNet

class Predictor:
    def __init__(self, board=None):
        self.depth = 10

        if board: self.board = chess.Board()
        else: self.board = board
        
    def find_best_move(self):
        move_list = list(self.board.legal_moves)
        turn = 0

        if self.board.turn == chess.WHITE: turn = 1
        else: turn = -1

        return self.minimax(self.depth, turn)


    def minimax(self, k, turn):
        if k==0:
            return 1


