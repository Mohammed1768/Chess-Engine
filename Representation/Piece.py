from .Matrix import Matrix

class Piece:
    def __init__(self, color, mat: Matrix = None):
        self.color = color
        self.matrix = mat if mat is not None else Matrix()
    def occupies_square(self, x, y):
        return self.matrix.occupied(x,y)

class Pawn(Piece):
    def setup(self, color):
        row = 2 if color==1 else 7
        for c in 'abcdefgh':
            self.pieces.insert(c,row)

class Knight(Piece):
    WHITE_START_POSITIONS = [('b', 1), ('g', 1)]  # White
    BLACK_START_POSITIONS = [('b', 8), ('g', 8)]  # Black

    def setup(self):
        for file, rank in (self.WHITE_START_POSITIONS if self.color else self.BLACK_START_POSITIONS):
            self.pieces.insert(file, rank)

class Bishop(Piece):
    WHITE_START_POSITIONS = [('c', 1), ('f', 1)]  # White
    BLACK_START_POSITIONS = [('c', 8), ('f', 8)]  # Black

    def setup(self):
        for file, rank in (self.WHITE_START_POSITIONS if self.color else self.BLACK_START_POSITIONS):
            self.pieces.insert(file, rank)


class Rook(Piece):
    WHITE_START_POSITIONS = [('a', 1), ('h', 1)]  # White
    BLACK_START_POSITIONS = [('a', 8), ('h', 8)]  # Black

    def setup(self):
        for file, rank in (self.WHITE_START_POSITIONS if self.color else self.BLACK_START_POSITIONS):
            self.pieces.insert(file, rank)

class Queen(Piece):
    WHITE_START_POSITIONS = [('d', 1)]  # White
    BLACK_START_POSITIONS = [('d', 8)]  # Black

    def setup(self):
        for file, rank in (self.WHITE_START_POSITIONS if self.color else self.BLACK_START_POSITIONS):
            self.pieces.insert(file, rank)


class King(Piece):
    WHITE_START_POSITIONS = [('e', 1)]  # White
    BLACK_START_POSITIONS = [('e', 8)]  # Black

    def setup(self):
        for file, rank in (self.WHITE_START_POSITIONS if self.color else self.BLACK_START_POSITIONS):
            self.pieces.insert(file, rank)
