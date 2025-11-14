# include <bits/stdc++.h>

typedef long long ll;
using namespace std;

int main() {
    ifstream infile("tactic_evals.csv");
    ofstream evals_out("evaluations.csv");
    ofstream mate_out("mate.csv");

    evals_out << "FEN,eval\n";
    mate_out << "FEN,eval\n";

    string line;
    getline(infile, line); 

    ll idx = 0;
    while (getline(infile, line)) {
        if (idx % 1000 == 0) cout << idx << "\n";
        idx++;

        stringstream ss(line);
        string fen, eval;

        if (!getline(ss, fen, ',')) continue;
        if (!getline(ss, eval)) continue;

        if (!eval.empty() && eval[0] == '#') {
            if (eval.size() > 1 && eval[1] == '+')  mate_out << fen << ",1\n";
            else mate_out << fen << ",-1\n";
        } 
        else evals_out << fen << "," << eval << "\n";
    }

    cout << "DONE\n";
}
