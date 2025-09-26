from Board import Board
from Representation.Piece import *

class Move:
    """
    color = -1 for black/ 1 for white
    piece = {0:pawn, 1:knight, 2:bishop, 3:rook, 4:queen, 5:king}
    f = from, Ex: e,4
    t = to, Ex: e4
    flag = "normal" | "capture" | "promotion" | "castle" | "enpassant"
    """
    piece_names = ["pawn", "knight", "bishop", "rook", "queen", "king"]

    def __init__(self, color, piece, f, t, flag):
        self.piece = piece
        self.color = color
        self.f = f
        self.t = t
        self.flag = flag
        self.promotion = None
    def __init__(self, color, piece, f, t, flag, promotion):
        self.piece = piece
        self.color = color
        self.f = f
        self.t = t
        self.flag = flag
        self.promotion = promotion


    def display_move(self):
        color_str = "White" if self.color == 1 else "Black"
        piece_str = self.piece_names[self.piece]
        move_str = f"{color_str} {piece_str} {self.f[0]}{self.f[1]} → {self.t[0]}{self.t[1]} ({self.flag})"
        print(move_str)



class Move_Generator:
    def __init__(self, board :Board):
        self.board = board
        self.last = 0

    def generate_moves(self, color):
        pawns,knights,bishops,rook,queens,king = self.board.pieces[:6] if color==1 else self.board.pieces[6:]
        
        p = self.pawn(pawns.matrix, color)
        p += self.knight(knights.matrix, color)
        p += self.bishop(bishops.matrix, color)
        p += self.rook(rook.matrix, color)
        p += self.queen(queens.matrix, color)
        p += self.king(king, rook, color)
        


    def legal(self, move: Move):
        """
        to do
        given a board state and a move
        return if performing the following move is legal or not
        
        """
        pass            
        

    def pawn(self, mat:Matrix, color):
        moves = []

        cols = "abcdefgh"
        for i in range(1,9):
            for j in range(8):
                c = cols[j]

                # empty square
                if not mat.occupied(c, i):
                    continue

                """check for captures"""
                # captures to the right
                if j < 7:
                    if color==1 and self.board.occupied_by_black(cols[j+1], i+1):
                        if i<7:  moves.append(Move(1, 0, (c,i),(cols[j+1],i+1), "capture"))
                        else: 
                            for d in [1,2,3,4]:
                                moves.append(Move(1, 0, (c,i),(cols[j+1],i+1), "promotion", d))

                    if color==-1 and self.board.occupied_by_white(cols[j+1], i-1):
                        if i>2: moves.append(Move(-1, 0, (c,i),(cols[j+1],i-1), "capture"))
                        else: 
                            for d in [1,2,3,4]:
                                moves.append(Move(-1, 0, (c,i),(cols[j+1],i-1), "promotion", d))
                
                # captures to the left
                if j > 0:
                    if color==1 and self.board.occupied_by_black(cols[j-1], i+1):
                        if i<7: moves.append(Move(1, 0, (c,i),(cols[j-1],i+1), "capture"))
                        else: 
                            for d in [1,2,3,4]:  moves.append(Move(1, 0, (c,i),(cols[j-1],i+1), "promotion", d))

                    if color==-1 and self.board.occupied_by_white(cols[j-1], i-1):
                        if i>2: moves.append(Move(-1, 0, (c,i),(cols[j-1],i-1), "capture"))
                        else: 
                            for d in [1,2,3,4]: moves.append(Move(-1, 0, (c,i),(cols[j-1],i-1), "promotion", d))
                
                # check for en-passant
                if self.last and self.last.piece==0:
                    d1,k1 = self.last.f
                    d2,k2 = self.last.t

                    if d1==d2 and abs(k1-k2)==2 and k2==i and abs(cols.index(c)-cols.index(d2))==1:
                        moves.append(Move(color, 0, (c,i), (d2, i+color),"enpassant"))

                if self.board.occupied_by_any(c,i+color):
                    continue

                if (i+color!=0 and i+color!=7): moves.append(Move(color, 0, (c,i), (c,i+color), "normal"))
                else: 
                    for d in [1,2,3,4]: moves.append(Move(color, 0, (c,i), (c,i+color), "promotion", d))

                if color==1 and i == 2 and (not self.board.occupied_by_any(c, i+2)):
                    moves.append(Move(color, 0, (c,2), (c,4), "normal"))
                if color==-1 and i == 7 and (not self.board.occupied_by_any(c, i-2)):
                    moves.append(Move(color, 0, (c,7), (c,5), "normal"))
        return moves
    def knight(self, mat:Matrix, color):
        moves = []

        cols = "abcdefgh"
        for i in range(1,9):
            for j in range(8):
                c = cols[j]
                if not mat.occupied(c, i):
                    continue
                
                for a in [1,2]:

                    for o1 in ['0', '1']:
                        for o2 in ['1', '0']:
                            b = 3 - a

                            new_i = i+a if o1=='0' else i-a
                            new_j = j+b if o2=='0' else j-b

                            if not (1<=new_i<=8 and 0<=new_j<=7):
                                continue 
                            
                            if color==1 and self.board.occupied_by_white(cols[new_j], new_i): continue
                            elif color==-1 and self.board.occupied_by_black(cols[new_j], new_i): continue
                            elif color==1 and self.board.occupied_by_black(cols[new_j], new_i): 
                                moves.append(Move(color, 1, (cols[j], i), (cols[new_j], new_i), "capture"))
                            elif color==-1 and self.board.occupied_by_white(cols[new_j], new_i): 
                                moves.append(Move(color, 1, (cols[j], i), (cols[new_j], new_i), "capture"))
                            else: moves.append(Move(color, 1, (cols[j], i), (cols[new_j], new_i), "normal"))
        return moves
    def bishop(self, mat:Matrix, color):
        moves = []

        cols = "abcdefgh"
        for i in range(1,9):
            for j in range(8):
                c = cols[j]

                if not mat.occupied(c, i):
                    continue

                DX = [1,1,-1,-1]
                DY = [-1,1,-1,1]
                active = [1, 1, 1, 1]
                diag = 1

                while diag<8:
                    for l in range(4):
                        if not active[l]:
                            continue
                        dx,dy = DX[l], DY[l]

                        new_i = i + diag*dx
                        new_j = j + diag*dy

                        if new_i>8 or new_i<1 or new_j>7 or new_j<0:
                            active[l] = 0
                            continue
                        
                        if color == 1:
                            if self.board.occupied_by_white(cols[new_j], new_i):
                                active[l] = 0
                                continue
                            if self.board.occupied_by_black(cols[new_j], new_i):
                                moves.append(Move(color, 2, (c, i), (cols[new_j], new_i), "capture"))
                                active[l] = 0
                        if color == -1:
                            if self.board.occupied_by_black(cols[new_j], new_i):
                                active[l] = 0
                                continue
                            if self.board.occupied_by_white(cols[new_j], new_i):
                                moves.append(Move(color, 2, (c, i), (cols[new_j], new_i), "capture"))
                                active[l] = 0

                        if not self.board.occupied_by_any(cols[new_j], new_i):
                            moves.append(Move(color, 2, (c, i), (cols[new_j], new_i), "normal"))
                    diag+=1
      
        return moves
    def rook(self, mat:Matrix, color):
        moves = []
        cols = "abcdefgh"

        for i in range(1,9):
            for j in range(8):
                c = cols[j]
                if not mat.occupied(c, i):
                    continue

                active = [1,1,1,1]
                dxdy = [1,-1, 1, -1]

                for len in range(1,8):
                    for k in range(4):
                        if not active[k]: continue

                        dx = dxdy[k]*len, dy=0
                        if k>=2: dx,dy = dy,dx

                        i_new = i + dx
                        j_new = j + dy

                        if i_new<1 or i_new>8 or j_new<0 or j_new>7:
                            active[k] = 0
                            continue
                        if color==1 and self.board.occupied_by_white(cols[j_new], i_new):
                            active[k] = 0
                            continue
                        if color==-1 and self.board.occupied_by_black(cols[j_new], i_new):
                            active[k] = 0
                            continue
                        
                        if not self.board.occupied_by_any(cols[j_new], i_new):
                            moves.append(Move(color, 3, (c, i), (cols[j_new], i_new), "normal"))
                        else:
                            moves.append(Move(color, 3, (c, i), (cols[j_new], i_new), "capture"))
                            active[k] = 0
        return moves
    def queen(self, mat:Matrix, color):
        return self.rook(mat, color) + self.bishop(mat, color)
    def king(self, king:King, rooks:Rook, color):
        moves = []      
        mat = king.pieces       
        col = "abcdefgh"

        DX = [1,-1,0]
        DY = [1,-1,0]

        for i in range(1,9):
            for j in range(8):
                c = col[j]

                if not mat.occupied(c,i):
                    continue
                
                for dx in DX:
                    for dy in DY:
                        if dx == dy == 0:
                            continue

                        new_i = i + dx
                        new_j = j + dy
                        if new_i>8 or new_i<1 or new_j<0 or new_j>7:
                            continue

                        if color==1 and self.board.occupied_by_white(col[new_j], new_i):
                            continue
                        elif color==-1 and self.board.occupied_by_black(col[new_j], new_i):
                            continue
                        elif self.board.occupied_by_any(col[new_j], new_i):
                            moves.append(Move(color, 5, (col[j], i), (col[new_j], new_i), "capture"))
                        else: moves.append(Move(color, 5, (col[j], i), (col[new_j], new_i), "normal"))

        row = 8 if color==-1 else 1
        if not king.moved:
            if not rooks.moved_l and not (self.board.occupied_by_any('d',row) or self.board.occupied_by_any('c',row)):
                moves.append(Move(color, 5, ('e',row), ('c',row), "castle"))    
            if not rooks.moved_r and not (self.board.occupied_by_any('f',row) or self.board.occupied_by_any('g',row)):
                moves.append(Move(color, 5, ('e',row), ('g',row), "castle"))    
        return moves



b = Board()
b.initialize()

b.white_pieces[0].pieces.move('e', 2, 'e', 5)
b.black_pieces[0].pieces.move('d', 7, 'd', 5)

g = Move_Generator(b)

g.white()