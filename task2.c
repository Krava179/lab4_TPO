#include "mpi.h"
#include <stdio.h>
#include <stdlib.h>

#define NRA 62          /* number of rows in matrix A */
#define NCA 15          /* number of columns in matrix A */
#define NCB 7           /* number of columns in matrix B */
#define MASTER 0        /* taskid of first task */
#define FROM_MASTER 1   /* setting a message type */
#define FROM_WORKER 2   /* setting a message type */

int main (int argc, char *argv[]) {
    int numtasks,
        taskid,
        numworkers,
        source,
        dest,
        rows,
        averow, extra, offset,
        i, j, k, rc;

    double a[NRA][NCA],
           b[NCA][NCB],
           c[NRA][NCB];

    MPI_Status  status;
    MPI_Request request; /* дескриптор неблокуючої операції */

    MPI_Init(&argc, &argv);
    MPI_Comm_size(MPI_COMM_WORLD, &numtasks);
    MPI_Comm_rank(MPI_COMM_WORLD, &taskid);

    if (numtasks < 2) {
        printf("Need at least two MPI tasks. Quitting...\n");
        MPI_Abort(MPI_COMM_WORLD, rc);
        exit(1);
    }

    numworkers = numtasks - 1;

    /**************************** master task ************************************/
    if (taskid == MASTER) {
        printf("mpi_mm_nonblocking has started with %d tasks.\n", numtasks);

        /* Initialize matrices */
        for (i = 0; i < NRA; i++)
            for (j = 0; j < NCA; j++)
                a[i][j] = 10;
        for (i = 0; i < NCA; i++)
            for (j = 0; j < NCB; j++)
                b[i][j] = 10;

        /* Масиви дескрипторів для відправки всім воркерам */
        MPI_Request send_requests[numworkers * 4];
        int         offsets[numworkers];
        int         rows_arr[numworkers];
        int         req_idx = 0;

        averow = NRA / numworkers;
        extra  = NRA % numworkers;
        offset = 0;

        /* Неблокуюча відправка даних усім воркерам одночасно */
        for (dest = 1; dest <= numworkers; dest++) {
            rows = (dest <= extra) ? averow + 1 : averow;
            offsets[dest-1]  = offset;
            rows_arr[dest-1] = rows;

            printf("Sending %d rows to task %d offset=%d\n", rows, dest, offset);

            MPI_Isend(&offsets[dest-1], 1, MPI_INT,
                      dest, FROM_MASTER, MPI_COMM_WORLD, &send_requests[req_idx++]);
            MPI_Isend(&rows_arr[dest-1], 1, MPI_INT,
                      dest, FROM_MASTER, MPI_COMM_WORLD, &send_requests[req_idx++]);
            MPI_Isend(&a[offset][0], rows * NCA, MPI_DOUBLE,
                      dest, FROM_MASTER, MPI_COMM_WORLD, &send_requests[req_idx++]);
            MPI_Isend(&b, NCA * NCB, MPI_DOUBLE,
                      dest, FROM_MASTER, MPI_COMM_WORLD, &send_requests[req_idx++]);

            offset += rows;
        }

        /* Чекаємо завершення всіх відправок */
        MPI_Waitall(req_idx, send_requests, MPI_STATUSES_IGNORE);

        /* Неблокуючий прийом результатів від усіх воркерів */
        MPI_Request recv_requests[numworkers * 3];
        int recv_offsets[numworkers];
        int recv_rows[numworkers];
        req_idx = 0;

        for (source = 1; source <= numworkers; source++) {
            MPI_Irecv(&recv_offsets[source-1], 1, MPI_INT,
                      source, FROM_WORKER, MPI_COMM_WORLD, &recv_requests[req_idx++]);
            MPI_Irecv(&recv_rows[source-1], 1, MPI_INT,
                      source, FROM_WORKER, MPI_COMM_WORLD, &recv_requests[req_idx++]);
            MPI_Irecv(&c[offsets[source-1]][0], rows_arr[source-1] * NCB, MPI_DOUBLE,
                      source, FROM_WORKER, MPI_COMM_WORLD, &recv_requests[req_idx++]);
        }

        /* Чекаємо отримання всіх результатів */
        MPI_Waitall(req_idx, recv_requests, MPI_STATUSES_IGNORE);

        for (source = 1; source <= numworkers; source++)
            printf("Received results from task %d\n", source);

        /* Print results */
        printf("****\n");
        printf("Result Matrix:\n");
        for (i = 0; i < NRA; i++) {
            printf("\n");
            for (j = 0; j < NCB; j++)
                printf("%6.2f ", c[i][j]);
        }
        printf("\n********\n");
        printf("Done.\n");
    }

    /**************************** worker task ************************************/
    else {
        MPI_Request recv_reqs[4];

        /* Неблокуючий прийом усіх даних від MASTER */
        MPI_Irecv(&offset,    1,          MPI_INT,    MASTER, FROM_MASTER, MPI_COMM_WORLD, &recv_reqs[0]);
        MPI_Irecv(&rows,      1,          MPI_INT,    MASTER, FROM_MASTER, MPI_COMM_WORLD, &recv_reqs[1]);
        MPI_Irecv(&a,         NRA * NCA,  MPI_DOUBLE, MASTER, FROM_MASTER, MPI_COMM_WORLD, &recv_reqs[2]);
        MPI_Irecv(&b,         NCA * NCB,  MPI_DOUBLE, MASTER, FROM_MASTER, MPI_COMM_WORLD, &recv_reqs[3]);

        /* Чекаємо отримання всіх даних перед обчисленням */
        MPI_Waitall(4, recv_reqs, MPI_STATUSES_IGNORE);

        /* Matrix multiply */
        for (k = 0; k < NCB; k++)
            for (i = 0; i < rows; i++) {
                c[i][k] = 0.0;
                for (j = 0; j < NCA; j++)
                    c[i][k] += a[i][j] * b[j][k];
            }

        /* Неблокуюча відправка результатів MASTER */
        MPI_Request send_reqs[3];
        MPI_Isend(&offset,   1,          MPI_INT,    MASTER, FROM_WORKER, MPI_COMM_WORLD, &send_reqs[0]);
        MPI_Isend(&rows,     1,          MPI_INT,    MASTER, FROM_WORKER, MPI_COMM_WORLD, &send_reqs[1]);
        MPI_Isend(&c,        rows * NCB, MPI_DOUBLE, MASTER, FROM_WORKER, MPI_COMM_WORLD, &send_reqs[2]);

        MPI_Waitall(3, send_reqs, MPI_STATUSES_IGNORE);
    }

    MPI_Finalize();
    return 0;
}
