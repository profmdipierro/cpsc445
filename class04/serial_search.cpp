#include <vector>
#include <set>
#include <iostream>

using namespace std;

set<int> search(const vector<int>& data, int x, int offset) {
  set<int> results;
  for(int i=0; i<data.size(); i++) {
    if (data[i] == x) {
      results.insert(i + offset);
    }
  }
  return results;
}



int main(int argc, char** argv) {

  int N = 10;
  vector<int> data(N);
  int x;
  set<int> results;

  for(int i=0; i<N; i++) {
    data[i] = i  %  4;
  }
  x = 2;
    
  results = search(data, x);

  for(int element : results) {
    cout << element << endl;
  }
  
  return 0;
}
