


vector<vector<int>> board;

int get(const vector<vector<int>> & board, i, j) {
  if (i<0|| i>=board.size()) {
    return 0;
  }
  if (j<0 || j>=board[i].size()) {
    return 0;
  }
  return board[i][j];
}

int new_value(const vector<vector<int>> & board, i, j) {
  int s =
    get(board, i-1, j-1) + get(board, i-1, j) + get(board, i-1, j+1)
    get(board, i, j-1) + 0 + get(board, i, j+1)
    get(board, i+1, j-1) + get(board, i+1, j) + get(board, i+1, j+1);
  int t = get(board, i, j);
  // ...
  return     
}

void step(vector<vector<int>> & input,
          vector<vector<int>> & output, int steps) {
  vector<vector<int>> * a = &input;
  vector<vector<int>> * b = &output;
  for step {
      for i { // parallelize this loop
	  for j {
            (*b)[i][j] = new_balue(*a, i, j)
	 }
      }
    swap(a,b)
  }
}


// life.exe a b c d
int main(int argc, char**argv) {
  argc == 5;
  argv[0] == "life.exe";
  argv[1] == "a";
  ..
  argv[4] == "d";
    
}
