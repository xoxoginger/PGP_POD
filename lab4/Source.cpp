#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <iostream>
#include <vector>
#include <ctime>
using namespace std;

int main(int argc, char* argv[]) {

    int N;
    cin >> N;

    vector<vector<double> > A(N, vector<double>(N, 0));


    for (int i = 0; i < N; i++)
    {
        for (int j = 0; j < N; j++)
            //A[i][j] = (double)rand() * 0.1 + 0.3; - tests
            cin >> A[i][j];//2 * (double)rand() / RAND_MAX - 1; // uniform random variables in [-1, 1)
    }

    vector<int> P(N, 0);
    for (int i = 0; i < N; i++)
        P[i] = i;

    unsigned int start = clock(); //- tests

    // outer loop over diagonal pivots
    for (int i = 0; i < N; i++) {

        // inner loop to find the largest pivot
        int idx = i; //maxPivot
        for (int k = i; k < N; ++k)
            if (fabs(A[k][i]) > fabs(A[i][i]))
            {
                idx = k;
            }

        P[i] = idx;
        // swap rows
        if (idx != i) {
            //swap(P[idx], P[i]);

            for (int k = i; k < N; k++)
                swap(A[i][k], A[idx][k]);
        }


        for (int k = i + 1; k < N; k++) { // iterate down rows
          // lower triangle factor is not zeroed out, keep it for L
            A[k][i] /= A[i][i];

            for (int j = i + 1; j < N; j++) // iterate across rows
              // subtract off lower triangle factor times pivot row
                A[k][j] = A[k][j] - A[i][j] * A[k][i];
        }
    }
    unsigned int end = clock(); // - tests
    unsigned int res = end - start;
    cout << res;


    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++)
            cout << "\t" << A[i][j];
        cout << endl;
    }

    cout << endl;

    for (int i = 0; i < N; i++)
        cout << " " << P[i];
    cout << endl;

    return 0;
}