
class Matrix:
    def __init__(self, board=None):
        if board is None: self.board = [[0 for _ in range(8)] for _ in range(8)]
        else: self.board = board


    # given example 'e', 4. return the index (i,j) of the board of the square
    def encode(x:chr, y:int):
        str = 'abcdefgh'
        return str.find(x), 8-y

    # occupied('e', 4) => return wherther or not the square is occupied
    def occupied(self,x,y):     
        i,j = self.encode(x,y) 
        return self.board[i][j]

    # insert('e',4) => insert a piece onto the given square
    def insert(self, x, y):
        i,j = self.encode(x,y)
        self.board[i][j] = 1

    # delete('e',4) => delete a piece currently on the given square
    def delete(self, x, y):
        i,j = self.encode(x,y)
        self.board[i][j] = 0

    # combination of insert and delete
    def move(self, x, y):
        i,j = self.encode(x,y)
        if not self.board[i][j]:
            raise Exception("Tried to move a piece from an empty square")
        self.delete(x,y)
        self.insert(x,y)

