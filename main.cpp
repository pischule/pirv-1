#include <iostream>
#include <omp.h>
#include <stdlib.h>
#include <vector>
#include <cstdlib>
#include <iomanip>

using namespace std;

int matrix_size = 0;
int block_size = 0;
vector<int> a;
vector<int> b;
int omp_mode = 0;

int get_random_number() {
    return rand() % 200 - 100;
}

vector<int> get_random_matrix() {
    vector<int> v(matrix_size * matrix_size);
    for (auto &item: v) {
        item = get_random_number();
    }
    return v;
}

bool parse_arguments(int argc, char *argv[]) {
    if (argc != 4) {
        return false;
    }
    matrix_size = atoi(argv[1]);
    block_size = atoi(argv[2]);
    omp_mode = atoi(argv[3]);
    return matrix_size > 0 &&
           block_size > 0 && block_size <= matrix_size
           && (matrix_size % block_size == 0)
           && omp_mode >= 0 && omp_mode <= 3;
}

void initialize_matrices() {
    a = get_random_matrix();
    b = get_random_matrix();
}

inline int pos(int y, int x) {
    return y * matrix_size + x;
}

vector<int> simple_multiplication() {
    vector<int> c(matrix_size * matrix_size, 0);
    for (int row = 0; row < matrix_size; ++row) {
        for (int col = 0; col < matrix_size; ++col) {
            for (int inner = 0; inner < matrix_size; inner++) {
                c[pos(row, col)] += a[pos(row, inner)] * b[pos(inner, col)];
            }
        }
    }
    return c;
}

vector<int> block_multiplication() {
    vector<int> c(matrix_size * matrix_size, 0);
    int number_of_blocks = matrix_size / block_size;
#pragma omp parallel for if (omp_mode == 1)
    for (int bi = 0; bi < number_of_blocks; ++bi)
#pragma omp parallel for if (omp_mode == 2)
            for (int bj = 0; bj < number_of_blocks; ++bj)
                for (int bk = 0; bk < number_of_blocks; ++bk)
                    for (int i = 0; i < block_size; i++)
                        for (int j = 0; j < block_size; j++)
                            for (int k = 0; k < block_size; k++)
                                c[pos(bi * block_size + i, bj * block_size + j)] +=
                                        a[pos(bi * block_size + i, bk * block_size + k)]
                                        * b[pos(bk * block_size + k, bj * block_size + j)];
    return c;
}

void multiplication_check() {
    initialize_matrices();
    vector<int> c1 = block_multiplication();
    vector<int> c2 = simple_multiplication();
    if (c1 != c2) {
        throw "multiplication error";
    }
}

int main(int argc, char *argv[]) {
    if (!parse_arguments(argc, argv)) return -1;
    initialize_matrices();

    double time_start = omp_get_wtime();
    block_multiplication();
    double time_end = omp_get_wtime();
    cout << fixed << setprecision(6) << time_end - time_start << endl;
    return 0;
}