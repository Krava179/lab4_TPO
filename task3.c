#include "mpi.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define MASTER      0
#define FROM_MASTER 1
#define FROM_WORKER 2
#define MAX_N       1000

/* Однопотокове множення (для вимірювання T1) */
void sequential_multiply(double *a, double *b, double *c, int nra, int nca, int ncb) {
    int i, j, k;
    for (i = 0; i < nra; i++)
        for (k = 0; k < ncb; k++) {
            c[i * ncb + k] = 0.0;
            for (j = 0; j < nca; j++)
                c[i * ncb + k] += a[i * nca + j] * b[j * ncb + k];
        }
}

/* Блокуюче множення */
double blocking_multiply(int nra, int nca, int ncb,
                         int taskid, int numtasks,
                         double *a, double *b, double *c) {
    int numworkers = numtasks - 1;
    int rows, averow, extra, offset, dest, source, i, j, k;
    MPI_Status status;
    double t_start, t_end;

    MPI_Barrier(MPI_COMM_WORLD);
    t_start = MPI_Wtime();

    if (taskid == MASTER) {
        averow = nra / numworkers;
        extra  = nra % numworkers;
        offset = 0;
        for (dest = 1; dest <= numworkers; dest++) {
            rows = (dest <= extra) ? averow + 1 : averow;
            MPI_Send(&offset,          1,          MPI_INT,    dest, FROM_MASTER, MPI_COMM_WORLD);
            MPI_Send(&rows,            1,          MPI_INT,    dest, FROM_MASTER, MPI_COMM_WORLD);
            MPI_Send(a + offset * nca, rows * nca, MPI_DOUBLE, dest, FROM_MASTER, MPI_COMM_WORLD);
            MPI_Send(b,                nca * ncb,  MPI_DOUBLE, dest, FROM_MASTER, MPI_COMM_WORLD);
            offset += rows;
        }
        for (source = 1; source <= numworkers; source++) {
            MPI_Recv(&offset, 1,         MPI_INT,    source, FROM_WORKER, MPI_COMM_WORLD, &status);
            MPI_Recv(&rows,   1,         MPI_INT,    source, FROM_WORKER, MPI_COMM_WORLD, &status);
            MPI_Recv(c + offset * ncb, rows * ncb, MPI_DOUBLE, source, FROM_WORKER, MPI_COMM_WORLD, &status);
        }
    } else {
        MPI_Recv(&offset, 1,            MPI_INT,    MASTER, FROM_MASTER, MPI_COMM_WORLD, &status);
        MPI_Recv(&rows,   1,            MPI_INT,    MASTER, FROM_MASTER, MPI_COMM_WORLD, &status);
        double *la = malloc(rows * nca * sizeof(double));
        double *lb = malloc(nca  * ncb * sizeof(double));
        double *lc = malloc(rows * ncb * sizeof(double));
        MPI_Recv(la, rows * nca, MPI_DOUBLE, MASTER, FROM_MASTER, MPI_COMM_WORLD, &status);
        MPI_Recv(lb, nca  * ncb, MPI_DOUBLE, MASTER, FROM_MASTER, MPI_COMM_WORLD, &status);
        for (k = 0; k < ncb; k++)
            for (i = 0; i < rows; i++) {
                lc[i * ncb + k] = 0.0;
                for (j = 0; j < nca; j++)
                    lc[i * ncb + k] += la[i * nca + j] * lb[j * ncb + k];
            }
        MPI_Send(&offset, 1,          MPI_INT,    MASTER, FROM_WORKER, MPI_COMM_WORLD);
        MPI_Send(&rows,   1,          MPI_INT,    MASTER, FROM_WORKER, MPI_COMM_WORLD);
        MPI_Send(lc,      rows * ncb, MPI_DOUBLE, MASTER, FROM_WORKER, MPI_COMM_WORLD);
        free(la); free(lb); free(lc);
    }

    MPI_Barrier(MPI_COMM_WORLD);
    t_end = MPI_Wtime();
    return t_end - t_start;
}

/* Неблокуюче множення */
double nonblocking_multiply(int nra, int nca, int ncb,
                            int taskid, int numtasks,
                            double *a, double *b, double *c) {
    int numworkers = numtasks - 1;
    int rows, averow, extra, offset, dest, source, i, j, k;
    double t_start, t_end;

    MPI_Barrier(MPI_COMM_WORLD);
    t_start = MPI_Wtime();

    if (taskid == MASTER) {
        int *offsets  = malloc(numworkers * sizeof(int));
        int *rows_arr = malloc(numworkers * sizeof(int));
        MPI_Request *sreqs = malloc(numworkers * 4 * sizeof(MPI_Request));
        int req_idx = 0;

        averow = nra / numworkers;
        extra  = nra % numworkers;
        offset = 0;
        for (dest = 1; dest <= numworkers; dest++) {
            rows = (dest <= extra) ? averow + 1 : averow;
            offsets[dest-1]  = offset;
            rows_arr[dest-1] = rows;
            MPI_Isend(&offsets[dest-1],    1,          MPI_INT,    dest, FROM_MASTER, MPI_COMM_WORLD, &sreqs[req_idx++]);
            MPI_Isend(&rows_arr[dest-1],   1,          MPI_INT,    dest, FROM_MASTER, MPI_COMM_WORLD, &sreqs[req_idx++]);
            MPI_Isend(a + offset * nca,    rows * nca, MPI_DOUBLE, dest, FROM_MASTER, MPI_COMM_WORLD, &sreqs[req_idx++]);
            MPI_Isend(b,                   nca  * ncb, MPI_DOUBLE, dest, FROM_MASTER, MPI_COMM_WORLD, &sreqs[req_idx++]);
            offset += rows;
        }
        MPI_Waitall(req_idx, sreqs, MPI_STATUSES_IGNORE);

        MPI_Request *rreqs = malloc(numworkers * 3 * sizeof(MPI_Request));
        req_idx = 0;
        for (source = 1; source <= numworkers; source++) {
            MPI_Irecv(&offsets[source-1],  1,                    MPI_INT,    source, FROM_WORKER, MPI_COMM_WORLD, &rreqs[req_idx++]);
            MPI_Irecv(&rows_arr[source-1], 1,                    MPI_INT,    source, FROM_WORKER, MPI_COMM_WORLD, &rreqs[req_idx++]);
            MPI_Irecv(c + offsets[source-1] * ncb, rows_arr[source-1] * ncb, MPI_DOUBLE, source, FROM_WORKER, MPI_COMM_WORLD, &rreqs[req_idx++]);
        }
        MPI_Waitall(req_idx, rreqs, MPI_STATUSES_IGNORE);
        free(offsets); free(rows_arr); free(sreqs); free(rreqs);
    } else {
        MPI_Request rreqs[4];
        double *la = malloc(nra * nca * sizeof(double));
        double *lb = malloc(nca * ncb * sizeof(double));
        double *lc = malloc(nra * ncb * sizeof(double));
        MPI_Irecv(&offset, 1,           MPI_INT,    MASTER, FROM_MASTER, MPI_COMM_WORLD, &rreqs[0]);
        MPI_Irecv(&rows,   1,           MPI_INT,    MASTER, FROM_MASTER, MPI_COMM_WORLD, &rreqs[1]);
        MPI_Irecv(la,      nra * nca,   MPI_DOUBLE, MASTER, FROM_MASTER, MPI_COMM_WORLD, &rreqs[2]);
        MPI_Irecv(lb,      nca * ncb,   MPI_DOUBLE, MASTER, FROM_MASTER, MPI_COMM_WORLD, &rreqs[3]);
        MPI_Waitall(4, rreqs, MPI_STATUSES_IGNORE);
        for (k = 0; k < ncb; k++)
            for (i = 0; i < rows; i++) {
                lc[i * ncb + k] = 0.0;
                for (j = 0; j < nca; j++)
                    lc[i * ncb + k] += la[i * nca + j] * lb[j * ncb + k];
            }
        MPI_Request sreqs[3];
        MPI_Isend(&offset, 1,           MPI_INT,    MASTER, FROM_WORKER, MPI_COMM_WORLD, &sreqs[0]);
        MPI_Isend(&rows,   1,           MPI_INT,    MASTER, FROM_WORKER, MPI_COMM_WORLD, &sreqs[1]);
        MPI_Isend(lc,      rows * ncb,  MPI_DOUBLE, MASTER, FROM_WORKER, MPI_COMM_WORLD, &sreqs[2]);
        MPI_Waitall(3, sreqs, MPI_STATUSES_IGNORE);
        free(la); free(lb); free(lc);
    }

    MPI_Barrier(MPI_COMM_WORLD);
    t_end = MPI_Wtime();
    return t_end - t_start;
}

int main(int argc, char *argv[]) {
    int taskid, numtasks;
    MPI_Init(&argc, &argv);
    MPI_Comm_size(MPI_COMM_WORLD, &numtasks);
    MPI_Comm_rank(MPI_COMM_WORLD, &taskid);

    if (numtasks < 2) {
        printf("Need at least two MPI tasks.\n");
        MPI_Abort(MPI_COMM_WORLD, 1);
    }

    /* Розміри матриць для тестування */
    int sizes[]  = {62, 200, 500, 800};
    int nsizes   = 4;
    int p        = numtasks;

    if (taskid == MASTER) {
        printf("Processes: %d\n\n", p);
        printf("%-6s %-10s %-12s %-12s %-10s %-10s %-14s %-14s %-10s\n",
               "N", "Method", "T1 (s)", "Tp (s)", "S=T1/Tp", "E=S/p",
               "Tblock (s)", "Tnonbl (s)", "Speedup");
        printf("%s\n", "------------------------------------------------------------------------"
                       "------------------------------");
    }

    int s;
    for (s = 0; s < nsizes; s++) {
        int nra = sizes[s], nca = sizes[s], ncb = sizes[s];

        double *a = malloc(nra * nca * sizeof(double));
        double *b = malloc(nca * ncb * sizeof(double));
        double *c = malloc(nra * ncb * sizeof(double));

        /* Ініціалізація матриць на MASTER */
        if (taskid == MASTER) {
            int i, j;
            for (i = 0; i < nra; i++)
                for (j = 0; j < nca; j++)
                    a[i * nca + j] = 10.0;
            for (i = 0; i < nca; i++)
                for (j = 0; j < ncb; j++)
                    b[i * ncb + j] = 10.0;

            /* T1 - однопотоковий час */
            double t1_start = MPI_Wtime();
            sequential_multiply(a, b, c, nra, nca, ncb);
            double t1 = MPI_Wtime() - t1_start;

            /* Tp блокуючий */
            double tp_block = blocking_multiply(nra, nca, ncb, taskid, numtasks, a, b, c);
            /* Tp неблокуючий */
            double tp_nonbl = nonblocking_multiply(nra, nca, ncb, taskid, numtasks, a, b, c);

            double s_block = t1 / tp_block;
            double e_block = s_block / (p - 1);
            double s_nonbl = t1 / tp_nonbl;
            double e_nonbl = s_nonbl / (p - 1);

            printf("%-6d %-10s %-12.6f %-12.6f %-10.3f %-10.3f\n",
                   nra, "Blocking", t1, tp_block, s_block, e_block);
            printf("%-6d %-10s %-12.6f %-12.6f %-10.3f %-10.3f   "
                   "Nonblocking швидше на %.2fx\n",
                   nra, "Nonblock", t1, tp_nonbl, s_nonbl, e_nonbl,
                   tp_block / tp_nonbl);
            printf("\n");
        } else {
            /* Воркери просто беруть участь у колективних операціях */
            blocking_multiply(nra, nca, ncb, taskid, numtasks, a, b, c);
            nonblocking_multiply(nra, nca, ncb, taskid, numtasks, a, b, c);
        }

        free(a); free(b); free(c);
    }

    MPI_Finalize();
    return 0;
}
