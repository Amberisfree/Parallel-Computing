#include <stdio.h>                // needed for printing
#include <math.h>                // needed for tanh, used in init function
#include "params.h"                // model & simulation parameters

#include <omp.h>

void init(double u[N][N], double v[N][N]){
    double uhi, ulo, vhi, vlo;
    
    
    
    uhi = 0.5; ulo = -0.5; vhi = 0.1; vlo = -0.1;
    int i, j;
    
    #pragma omp parallel shared(N,u,v) private(i,j)
    {
        #pragma omp for nowait collapse(2)
        for (i=0; i < N; i++){
            for (j=0; j < N; j++){
                u[i][j] = ulo + (uhi-ulo)*0.5*(1.0 + tanh((i-N/2)/16.0));
            }
        }
        #pragma omp for nowait collapse(2)
        for (i=0; i < N; i++){
            for (j=0; j < N; j++){
                v[i][j] = vlo + (vhi - vlo) * 0.5 * (1.0 + tanh((j - N / 2) / 16.0));
            }
            
        }
    }/*-- End of parallel region --*/
        
}

void dxdt(double du[N][N], double dv[N][N], double u[N][N], double v[N][N]) {
    #pragma omp parallel for collapse(2) shared(N, du, dv, u, v) schedule(static)
    {
        for (int i = 0; i < N; i++) {
            for (int j = 0; j < N; j++) {
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
    }/*-- End of parallel region --*/
}

void step(double du[N][N], double dv[N][N], double u[N][N], double v[N][N]){
    int i,j;
    #pragma omp parallel shared(N,u,v,du,dv) private(i,j)
    {
        #pragma omp for  collapse(2) nowait /*-- Work-sharing loop 1 --*/
        for (i = 0; i < N; i++){
            for (j = 0; j < N; j++){
                u[i][j] += dt*du[i][j];
                
            }
        }
        
        #pragma omp for collapse(2) nowait/*-- Work-sharing loop 2 --*/
        for (i = 0; i < N; i++){
            for (j = 0; j < N; j++){
                
                v[i][j] += dt*dv[i][j];
            }
        }
        
    }/*-- End of parallel region --*/
}



double norm(double x[N][N]){
    double nrmx = 0.0;
    
    int i,j;
    #pragma omp parallel default(none) \
        shared(N,x,nrmx) private(i,j)
    {
        #pragma omp for nowait reduction(+:nrmx)
        for (int i = 0; i < N; i++){
            for (int j = 0; j < N; j++){
                nrmx += x[i][j]*x[i][j];
            }
        }
        
    }/*-- End of parallel region --*/

    return nrmx;
}

int main(int argc, char **argv) {
    double t = 0.0, nrmu, nrmv;
    double u[N][N], v[N][N], du[N][N], dv[N][N];

    FILE *fptr = fopen("part1.dat", "w");
    fprintf(fptr, "#t\t\tnrmu\t\tnrmv\n");

    double start = omp_get_wtime();

    // initialize the state
    init(u, v);

    // time-loop
    for (int k = 0; k < M; k++) {
        // track the time
        t = dt * k;

        // evaluate the PDE
        dxdt(du, dv, u, v);

        // update the state variables u,v
        step(du, dv, u, v);

        if (k % m == 0) {
            // calculate the norms
            nrmu = norm(u);
            nrmv = norm(v);
            printf("t = %2.1f\tu-norm = %2.5f\tv-norm = %2.5f\n", t, nrmu, nrmv);
            fprintf(fptr, "%f\t%f\t%f\n", t, nrmu, nrmv);
        }
    }

    double end = omp_get_wtime();
    double cpu_time_used = end - start;

    printf("CPU time used: %f seconds\n", cpu_time_used);

    fclose(fptr);
    return 0;
}
