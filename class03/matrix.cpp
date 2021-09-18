#include <iostream>
#include <vector>

using namespace std;

class Point {
public:
  double m[3];
  Point(double x=0, double y=0, double z=0) {
    m[0] = x;
    m[1] = y;
    m[2] = z;
  }
};

class Matrix {
public:
  double m[9];
  Matrix(double a00=1, double a01=0, double a02=0,
	 double a10=0, double a11=1, double a12=0,
	 double a20=0, double a21=0, double a22=1) {
    m[0] = a00;
    m[1] = a01;
    m[2] = a02;
    m[3] = a10;
    m[4] = a11;
    m[5] = a12;
    m[6] = a20;
    m[7] = a21;
    m[8] = a22;
  }
};

double operator*(const Point& p, const Point& q) {
  double result = 0;
  for(int c=0; c<3; c++) {
    result += p.m[c] * q.m[c];
  }
  return result;
}

void mul(double* q, const double* m, const double* p) {
  for(int r=0; r<3; r++) {
    q[r]=0;
    for(int c=0; c<3; c++) {
      q[r] += m[3*r + c]*p[c];  // q = m * p;
    }
  }
}

void mul(Point& q, const Matrix& M, const Point& p) {
  mul(q.m, M.m, p.m);
}

Point operator*(const Matrix& M, const Point& p) {
  Point q;
  mul(q.m, M.m, p.m);
  return q;
}

ostream& operator<<(ostream& os, const Point& p)
{
  os << "(" << p.m[0] << "," << p.m[1] << "," << p.m[2] << ")";
  return os;
}

void mul_task(vector<Point>& w, const Matrix& M, const vector<Point>& v,
	      size_t rank, size_t p) {
  size_t N = v.size();
  size_t size = (N / p) + ((rank < N % p)?1:0);
  size_t index_start = rank * (N / p) + min(rank, N % p);
  size_t index_end = index_start + size;
  cout
    << "My rank is " << rank
    << " start=" << index_start
    << " size=" << size
    << " end=" << index_end << endl;

  for(size_t i=index_start; i<index_end; i++) {
    mul(w[i], M, v[i]);
  }
}

int main(int argc, char ** argv) {

  int p = 4;
  const int N=23;
  vector<Point> v(N);
  vector<Point> w(N);

  // initialization
  Matrix M(1,0,0,0,1,0,0,0,1);
  
  for(size_t i=0; i<v.size(); i++) {
    v[i] = Point(i, i/10, i*i);
  }

  // computation
  for(size_t rank=0; rank<p; rank++) {
    mul_task(w, M, v, rank, p);
  }

  // output
  cout << w[3] << endl;
  return 0;
}
