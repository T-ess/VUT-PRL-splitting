#include <iostream>
#include <fstream>
#include <vector>
#include <cmath>
#include <mpi.h>

using std::cout;
using std::endl;

int main(int argc, char *argv[]) {
    MPI_Init(&argc, &argv);
    std::vector<uint8_t> Li, Ei, Gi;
    int rank, size, loc_size;
    uint8_t *global = NULL;
    uint8_t median;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    // read numbers from file
    if (rank == 0) {
        uint8_t num;
        std::ifstream fileIn ("numbers", std::ios::binary);
        fileIn.seekg(0, std::ios::end);
        size_t length = fileIn.tellg();
        fileIn.seekg(0, std::ios::beg);
        global = (uint8_t*)malloc(length * sizeof(uint8_t));
        fileIn.read(reinterpret_cast<char*>(global), length);
        loc_size = length/size;
        int midIndex = trunc((float)length/2 - 0.5);
        median = global[midIndex];
    }

    // each process gets the value of median and its count of numbers
    MPI_Bcast(&median, 1, MPI_UNSIGNED_CHAR, 0, MPI_COMM_WORLD);
    MPI_Bcast(&loc_size, 1, MPI_INT, 0, MPI_COMM_WORLD);
    uint8_t local[loc_size];

    // each process gets it's data
    MPI_Scatter(global, loc_size, MPI_UNSIGNED_CHAR, 
                &local, loc_size, MPI_UNSIGNED_CHAR, 
                0, MPI_COMM_WORLD);
    // split the values according to the median
    for (int i = 0; i < loc_size; i++) {
        if (local[i] == median) {
            Ei.push_back(local[i]);
        } else if (local[i] < median) {
            Li.push_back(local[i]);
        } else {
            Gi.push_back(local[i]);
        }
    }
    int Lsize = Li.size();
    int Esize = Ei.size();
    int Gsize = Gi.size();

    // create lists of sizes
    int Lsizes[size], Esizes[size], Gsizes[size];
    MPI_Gather( &Lsize, 1, MPI_INT,
                Lsizes, 1, MPI_INT,
                0, MPI_COMM_WORLD);
    MPI_Gather( &Esize, 1, MPI_INT,
                Esizes, 1, MPI_INT,
                0, MPI_COMM_WORLD);
    MPI_Gather( &Gsize, 1, MPI_INT,
                Gsizes, 1, MPI_INT,
                0, MPI_COMM_WORLD);

    // displacement calculation for Gatherv
    int Ldispls[size];
    int Edispls[size];
    int Gdispls[size];
    int Ltotal = Lsize;
    int Etotal = Esize;
    int Gtotal = Gsize;
    if (rank == 0) {
        Ldispls[0] = 0;
        Edispls[0] = 0;
        Gdispls[0] = 0;
        for (int i=1; i<size; i++) {
            Ldispls[i] = Ldispls[i-1] + Lsizes[i-1];
            Edispls[i] = Edispls[i-1] + Esizes[i-1];
            Gdispls[i] = Gdispls[i-1] + Gsizes[i-1];
            Ltotal += Lsizes[i];
            Etotal += Esizes[i];
            Gtotal += Gsizes[i];
        }
    }

    // calculate final sizes of arrays and create them
    MPI_Bcast(&Ltotal, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&Etotal, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&Gtotal, 1, MPI_INT, 0, MPI_COMM_WORLD);
    uint8_t *L = NULL;
    uint8_t *E = NULL;
    uint8_t *G = NULL;
    if (rank == 0) {
        L = (uint8_t*)malloc(Ltotal * sizeof(uint8_t));
        E = (uint8_t*)malloc(Etotal * sizeof(uint8_t));
        G = (uint8_t*)malloc(Gtotal * sizeof(uint8_t));
    }

    // gather data from processes
    MPI_Gatherv(Li.data(), Lsize, MPI_UNSIGNED_CHAR,
                L, Lsizes, Ldispls, MPI_UNSIGNED_CHAR,
                0, MPI_COMM_WORLD);

    MPI_Gatherv(Ei.data(), Esize, MPI_UNSIGNED_CHAR,
                E, Esizes, Edispls, MPI_UNSIGNED_CHAR,
                0, MPI_COMM_WORLD);

    MPI_Gatherv(Gi.data(), Gsize, MPI_UNSIGNED_CHAR,
                G, Gsizes, Gdispls, MPI_UNSIGNED_CHAR,
                0, MPI_COMM_WORLD);

    // print out results
    if (rank == 0) {
        cout << "[";
        for (int i = 0 ; i < Ltotal ; i++) {
            cout << (int)L[i];
            (i < Ltotal - 1) ? cout << "," : cout << "] < ";
        }
        if (Ltotal == 0) cout << "] < ";
        cout << "[";
        for (int i = 0 ; i < Etotal ; i++) {
            cout << (int)E[i];
            (i < Etotal - 1) ? cout << "," : cout << "] < ";
        }
        if (Etotal == 0) cout << "] < ";
        cout << "[";
        for (int i = 0 ; i < Gtotal ; i++) {
            cout << (int)G[i];
            (i < Gtotal - 1) ? cout << "," : cout << "]" << endl;
        }
        if (Gtotal == 0) cout << "]" << endl;
        free(global);
    }

    MPI_Finalize();
    return 0;
}
