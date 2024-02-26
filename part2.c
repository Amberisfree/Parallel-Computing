#include <stdio.h>				// needed for printing
#include <math.h>				// needed for tanh, used in init function
#include "params.h"				// model & simulation parameters
            
#include <mpi.h>


void halo_exchange(double array[N][N], int rank, int size, int start_column, int end_column) {
    // Buffer for sending and receiving halo values
    double send_left[N];
    double recv_left[N];
    double send_right[N];
    double recv_right[N];

    // Send left, receive right
    for (int i = 0; i < N; i++) {
        send_left[i] = array[i][start_column];
    }
    MPI_Sendrecv(send_left, N, MPI_DOUBLE, (rank - 1 + size) % size, 0,
                 recv_right, N, MPI_DOUBLE, (rank + 1) % size, 0,
                 MPI_COMM_WORLD, MPI_STATUS_IGNORE);

    for (int i = 0; i < N; i++) {
        array[i][end_column] = recv_right[i];
    }

    // Send right, receive left
    for (int i = 0; i < N; i++) {
        send_right[i] = array[i][end_column - 1];
    }
    MPI_Sendrecv(send_right, N, MPI_DOUBLE, (rank + 1) % size, 1,
                 recv_left, N, MPI_DOUBLE, (rank - 1 + size) % size, 1,
                 MPI_COMM_WORLD, MPI_STATUS_IGNORE);

    for (int i = 0; i < N; i++) {
        array[i][start_column - 1] = recv_left[i];
    }
}

void init(double u[N][N], double v[N][N]) {
    double uhi, ulo, vhi, vlo;
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    uhi = 0.5; ulo = -0.5; vhi = 0.1; vlo = -0.1;


    for (int i = 0; i < N; i++) {
        for (int j =  0  ; j < N ; j++) {
            u[i][j] = ulo + (uhi - ulo) * 0.5 * (1.0 + tanh((i - N / 2) / 16.0));
            v[i][j] = vlo + (vhi - vlo) * 0.5 * (1.0 + tanh((j - N / 2) / 16.0));
        }
    }
}





void dxdt(double du[N][N], double dv[N][N], double u[N][N], double v[N][N]) {
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    int columns_per_process = N / size;
    int start_column = rank * columns_per_process;
    int end_column = (rank == size - 1) ? N : start_column + columns_per_process;

    // Halo exchange
    halo_exchange(u, rank, size, start_column, end_column);
    halo_exchange(v, rank, size, start_column, end_column);

    // Perform computation
    for (int i = 0; i < N; i++) {
        for (int j = start_column; j < end_column; j++) {
            int up = (i == N - 1) ? i : (i + 1);
            int down = (i == 0) ? i : (i - 1);
            int left = (j == 0) ? j : (j - 1);
            int right = (j == N - 1) ? j : (j + 1);

            double lapu = u[up][j] + u[down][j] + u[i][left] + u[i][right] - 4.0 * u[i][j];
            double lapv = v[up][j] + v[down][j] + v[i][left] + v[i][right] - 4.0 * v[i][j];

            du[i][j] = DD * lapu + u[i][j] * (1.0 - u[i][j]) * (u[i][j] - b) - v[i][j];
            dv[i][j] = d * DD * lapv + c * (a * u[i][j] - v[i][j]);
        }
    }
}


void step(double du[N][N], double dv[N][N], double u[N][N], double v[N][N]){
    int size, rank;
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);


    for (int i = 0; i <N; i++){
        for (int j = 0; j < N; j++){

            u[i][j] += dt*du[i][j];
            v[i][j] += dt*dv[i][j];

        }
    }
}


double norm(double x[N][N]) {
    MPI_Status mpi_status;
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    double localSum = 0.0;
    int localCols = N / size;

    // Calculate the start and end indices for the local columns
    int startCol = rank * localCols;
    int endCol = (rank == size - 1) ? N : startCol + localCols;

    // Iterate over local rows and columns
    for (int i = 0; i < N; i++) {
        for (int j = startCol; j < endCol; j++) {
            localSum += x[i][j] * x[i][j];
        }
    }
  

    // Sum the local partial sums across all processes
    double globalSum;

    MPI_Allreduce(&localSum, &globalSum, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);

    return globalSum;
}






int main(int argc, char** argv){
    MPI_Init(&argc, &argv);
    int size, rank;
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    
    
	double t = 0.0, nrmu, nrmv;
	double u[N][N], v[N][N], du[N][N], dv[N][N];
	
	FILE *fptr = fopen("part2.dat", "w");
	fprintf(fptr, "#t\t\tnrmu\t\tnrmv\n");
	
    double start = MPI_Wtime();
        
	// initialize the state
	init(u, v);
    
	// time-loop
	for (int k=0; k < M; k++){
		// track the time
		t = dt*k;
		// evaluate the PDE
		dxdt(du, dv, u, v);
		// update the state variables u,v
		step(du, dv, u, v);
		if (k%m == 0){
			// calculate the norms
			nrmu = norm(u);
			nrmv = norm(v);
            // Check if nrmu and nrmv are not NaN
            if (!isnan(nrmu) && !isnan(nrmv)) {
                printf("t = %2.1f\tu-norm = %2.5f\tv-norm = %2.5f\n", t, nrmu, nrmv);
                fprintf(fptr, "%f\t%f\t%f\n", t, nrmu, nrmv);
            }
		}
	}
	
    
    double end = MPI_Wtime();
    double total = end - start;
    printf("time = %2.1f\n", total);
   
    MPI_Finalize();
	fclose(fptr);

	return 0;
}
