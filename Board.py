from Representation.Piece import *
import numpy as np

class Board:
    def __init__(self):
        self.pieces = [
            Pawn(+1), Knight(+1), Bishop(+1), Rook(+1), Queen(+1), King(+1),
            Pawn(-1), Knight(-1), Bishop(-1), Rook(-1), Queen(-1), King(-1)
        ]

    def setup(self):
        for piece in self.pieces: piece.setup()
    

    def occupied_by(self, x,y):      
        for piece in self.pieces:
            if piece.occupies_square(x,y):
                return 1
        return 0
    
    def to_planes(self):
        return np.stack([p.matrix.to_numpy() for p in self.pieces])  # (12,8,8)
        