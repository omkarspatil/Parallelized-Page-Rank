#include <stdio.h>
#include <stdlib.h>
#include "csr_reader.h"
#include <string.h>
#include <mpi.h>
#include <math.h>
#include <sys/time.h>
#include <time.h>

#define N 1000000000
int mark_colind[N];

#define CONVERGENCE_LIMIT 0.00001

void initializeRanks(double **answer, int n);
void printIntList(int * list, int n);
void printDoubleList(double * list, int n);
void CSRMatVec(double * value, int * colind, int *rbegin, double * x, int n, double **y, int offset);
void vector_sub(double ** result, double* vec_a, double* vec_b, int l);
void vector_add(double ** result, double* vec_a, double* vec_b, int l);
void vector_const_multiply(double ** result, double constant, double* vec_a, int l);
double vector_sum(double* vec_a, int l);
double infinity_norm(double* vec_a, int l);

typedef enum tags
{
  VALUES,
  RBEGIN,
  COLIND,
  VECTOR_X,
  OFFSET,
  X_OFFSET,
  N_ALL,
  N_PART,
  N_ZEROS,
  PRANK,
  RECV_COUNTS,
  RECV_DISPLS,
  RECV_FROM_OTHERS_COUNT,
  SEND_TO_OTHERS_COUNT,
  RECV_FROM_OTHERS_INDICES,
  SEND_TO_OTHERS_INDICES,
  X_PART,
  EXIT
} tags;

int main(int argc, char* argv[])
{
	double *values=NULL;
	int *colind=NULL;
	int *rbegin=NULL;
	int offset;
	int x_offset;
	int n;
	int n_a;
	int nzeros;
	int *recv_c;
	int *recv_d;
	int *to_receive_counts;
	int *to_send_counts;


	int exit;

	double *prank_part=NULL;
	double *prank_send_buffer=NULL;
	double *prank_full=NULL;

	double *gathered_result=NULL;

    double start_time, end_time;

  	struct timeval tz;
  	struct timezone tx;

  	MPI_Init(NULL, NULL);
  	int world_rank;
  	MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
  	int world_size;
  	MPI_Comm_size(MPI_COMM_WORLD, &world_size);


  	int *to_receive_values[world_size];
  	double *received_values[world_size];

	int *to_send_values[world_size];

  	float time_elapsed =0;

	if(world_rank == 0)
	{
		double *values_all=NULL;
		int *colind_all=NULL;
		int *rbegin_all=NULL;
		int n_all;
		int nzeros_all; 

		//Read the file and get the data to be distributed.
		readfile(argv[1],&n_all,&nzeros_all, &values_all,&colind_all,&rbegin_all);
		//printDoubleList(values_all,nzeros_all);
		//printIntList(colind_all,nzeros_all);
		//printIntList(rbegin_all,n_all+1);


		gettimeofday(&tz, &tx);
   		start_time = (double)tz.tv_sec + (double) tz.tv_usec / 1000000.0;
        

		double *prank_previous = NULL;
		double *prank;
		
		//Form initial rank vector to be distributed.
		initializeRanks(&prank, n_all);

		//printf("N : %d\n",n_all);
		int rows_per_process = n_all/world_size;
		int extra_work = n_all % world_size;
		int mod = extra_work;
		//printf("Rows per proc %d\n", rows_per_process);
		//printf("Extra work %d\n", extra_work);
		//printf("NALL %d\n", n_all);

		int recv_counts[world_size];
		int recv_displs[world_size];

		int receive_send_counts[world_size][world_size];
		for (int i = 0; i < world_size; i++)
  			for (int j = 0; j < world_size; j++)
   			   receive_send_counts[i][j] = 0;

		int* receive_send_values[world_size][world_size];


		int send_receive_counts[world_size][world_size];
		for (int i = 0; i < world_size; i++)
  			for (int j = 0; j < world_size; j++)
   			   send_receive_counts[i][j] = 0;

		int* send_receive_values[world_size][world_size];

		int start_indices[world_size];
		int n_zeros_parts[world_size];



		for(int i=0; i<world_size ;i++)
			{
				int start_index;
				int end_index;
				if(extra_work!=0)
				{
					////printf("Extra work %d\n", extra_work);
					start_index = i*(rows_per_process+1);
					end_index =(i+1)*(rows_per_process+1);
					extra_work--;	
				}
				else
				{
					start_index = i*(rows_per_process) + mod;
					end_index =(i+1)*(rows_per_process) + mod;
				}

				
				int n_part = (end_index - start_index);
				int n_zeros_part = rbegin_all[end_index] - rbegin_all[start_index]+ 1;
				//printf("N_part: %d, n_zeros_part: %d\n", n_part, n_zeros_part);
				//Send out the number of elements in each part 
				MPI_Request req1;
				MPI_Isend(&n_all, 1, MPI_INT, i, N_ALL, MPI_COMM_WORLD,&req1);
				if(i==0)
				{
					MPI_Recv(&n_a, 1, MPI_INT, 0, N_ALL, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
					//printf("Process %d received n all %d", world_rank, n_a);
				}

				MPI_Request req2;
				MPI_Isend(&n_part, 1, MPI_INT, i, N_PART, MPI_COMM_WORLD,&req2);
				if(i==0)
				{
					MPI_Recv(&n, 1, MPI_INT, 0, N_PART, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
					//printf("Process %d received n ", world_rank);
				}
			
				MPI_Request req3;
				MPI_Isend(&n_zeros_part, 1, MPI_INT, i, N_ZEROS, MPI_COMM_WORLD,&req3);
				if(i==0)
				{
					MPI_Recv(&nzeros, 1, MPI_INT, 0, N_ZEROS, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
					//printf("Process %d received nzeros ", world_rank);
				}
			
				
				////printf("RB %d,%d,%d\n", rbegin_all[start_index] , rbegin_all[end_index], rbegin_all[end_index] - rbegin_all[start_index]);

				////printDoubleList(values_all[rbegin_all[start_index]], rbegin_all[end_index] - rbegin_all[start_index] + 1);
				////printIntList(rbegin_all[rbegin_all[start_index]], end_index - start_index + 1);
				////printIntList(colind_all[rbegin_all[start_index]], rbegin_all[end_index] - rbegin_all[start_index] + 1);

				//Send out the partial values vector to each process
				//double* part_values = (double *) malloc(sizeof(double)*(n_zeros_part));
				//memcpy(part_values, values_all + rbegin_all[start_index], sizeof(double) * (n_zeros_part));

				MPI_Request req4;
				//printf("Process %d done with memcpy for values \n", world_rank);
				////printDoubleList(part_values, rbegin_all[end_index] - rbegin_all[start_index]+ 1);
				MPI_Isend(values_all + rbegin_all[start_index], n_zeros_part, MPI_DOUBLE, i, VALUES, MPI_COMM_WORLD, &req4);
				if(i==0)
				{
					values = (double*) malloc(sizeof(double)*nzeros);
					MPI_Recv(values, nzeros, MPI_DOUBLE, 0, VALUES, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
					//printf("Process %d received values", world_rank);
					//printDoubleList(values, nzeros);
				}
				
				MPI_Request req5;
				//Send out the partial rbegin vector to each process
				//int* part_rbegin = (int*) malloc(sizeof(int)*(n_part+1));
				//memcpy(part_rbegin, rbegin_all + start_index , sizeof(int) * (n_part + 1));
				////printIntList(part_rbegin, n_part+1);
				MPI_Isend(rbegin_all + start_index, n_part + 1, MPI_INT, i, RBEGIN, MPI_COMM_WORLD, &req5);
				if(i==0)
				{
					rbegin = (int*) malloc(sizeof(int)*n+1);
					MPI_Recv(rbegin, n+1, MPI_INT, 0, RBEGIN, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
					//printf("Process %d received rbegin ", world_rank);
					//printIntList(rbegin, n+1);
				}

				MPI_Request req6;
				//Send out the partial colindex vector to each process
				//int* part_colind = (int *) malloc(sizeof(int)*(n_zeros_part));
				//memcpy(part_colind, colind_all + rbegin_all[start_index], sizeof(int) * (n_zeros_part));
				MPI_Isend(colind_all + rbegin_all[start_index] , n_zeros_part, MPI_INT, i, COLIND, MPI_COMM_WORLD, &req6);
				if(i==0)
				{
					colind = (int*) malloc(sizeof(int)*nzeros);
					MPI_Recv(colind, nzeros, MPI_INT, 0, COLIND, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
					//printf("Process %d received colind ", world_rank);
					//printIntList(colind, nzeros);
				}
				////printIntList(part_colind, rbegin_all[end_index] - rbegin_all[start_index]+ 1);
				
				MPI_Request req7;
				//Send out the offset to each process
				int offset_p = rbegin_all[start_index]; 
				MPI_Isend(&offset_p, 1, MPI_INT, i, OFFSET, MPI_COMM_WORLD,&req7);
				if(i==0)
				{
					MPI_Recv(&offset, 1, MPI_INT, 0, OFFSET, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
					//printf("\nProcess %d received offset %d\n", world_rank, offset);
				}

				MPI_Request req8;
				//Send out the offset to each process
				int x_offset_p = start_index; 
				MPI_Isend(&x_offset_p, 1, MPI_INT, i, X_OFFSET, MPI_COMM_WORLD,&req7);
				if(i==0)
				{
					MPI_Recv(&x_offset, 1, MPI_INT, 0, X_OFFSET, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
					//printf("\nProcess %d received offset %d\n", world_rank, offset);
				}
				////printf("Start Index %d\n", start_index);
				////printf("End Index %d\n", end_index);
				////printf("Part length %d\n", n_part);
				start_indices[i] = start_index;
				n_zeros_parts[i] = n_zeros_part;

				recv_counts[i] = n_part;
				if(i==0)
				{
					recv_displs[i] = 0;
				}
				else
				{
					recv_displs[i] = recv_displs[i-1] + recv_counts[i-1];
				}


			}


	//printf("Starting to send init data\n");

	for(int i = 0;i <world_size ; i++)
	{	
		/*printf("S1\n");*/
		//int mark_colind[nzeros_all];
		for (int i = 0; i < nzeros_all; i++)
   			   mark_colind[i]= 0;
   		/*printf("S2\n");*/


		//printf("n_zeros_part %d : %d\n", i, n_zeros_parts[i]);
		for(int c = 0 ; c<n_zeros_parts[i]; c++)
		{
			//printf("S3\n");
			int cur_colind = *(colind_all + (rbegin_all[start_indices[i]] + c));
			//printf("colind for process %d : %d : idc %d\n", i, cur_colind, c);
			if(!mark_colind[cur_colind])
			{
				//printf("S4\n");				//
				//Find which process id has this index of the x vector.
				int peer_id;
				int assigned = 0;
				for(int s=0;s<world_size;s++)
				{
					//printf("S5\n");
					if(start_indices[s]>cur_colind)
					{
						//printf("S6\n");
						peer_id = s - 1;
						assigned = 1;
						break;
					}
				}
				if(!assigned)
				{
					//printf("S7\n");
					peer_id = world_size-1;
				}
				//printf("S8\n");
				//printf("colind for process %d : %d is to be recieved from peer %d\n", i, cur_colind,peer_id);

				mark_colind[cur_colind] = 1;
				//if(peer_id == i) continue;

				receive_send_counts[i][peer_id]++;
				send_receive_counts[peer_id][i]++;
			}
		}
	}

	for(int i=0;i<world_size;i++)
	{
		for(int j=0;j<world_size;j++)
		{
			//printf("Receive to send count %d to %d is %d \n", i, j, receive_send_counts[i][j]);
			receive_send_values[i][j] = (int*) malloc(sizeof(int) * receive_send_counts[i][j]);
			send_receive_values[i][j] = (int*) malloc(sizeof(int) * send_receive_counts[i][j]);
			/*for (int k = 0; k < receive_send_counts[i][j]; k++) 
				{
					receive_send_values[i][j][k]= 0;
				}*/
		}
	}

	//printf("After Starting to send init data\n");

	int receive_counters[world_size][world_size];
	int send_counters[world_size][world_size];

	for (int i = 0; i < world_size; i++)
	{
  			for (int j = 0; j < world_size; j++)
  			{
  				send_counters[i][j] =0;
   			   receive_counters[i][j] = 0;
  			}

	}



	for(int i = 0;i <world_size ; i++)
	{
		//int mark_colind[nzeros_all];
		

		for (int i = 0; i < nzeros_all; i++)
   			   mark_colind[i]= 0;

		//printf("n_zeros_part values %d : %d\n", i, n_zeros_parts[i]);
		for(int c = 0 ; c<n_zeros_parts[i]; c++)
		{
			int cur_colind = *(colind_all + (rbegin_all[start_indices[i]] + c));
			//printf("colind for process %d : %d : idc %d\n", i, cur_colind, c);
			if(mark_colind[cur_colind]==0)
			{
				//Find which process id has this index of the x vector.
				int peer_id;
				int assigned = 0;
				for(int s=0;s<world_size;s++)
				{
					if(start_indices[s]>cur_colind)
					{
						peer_id = s - 1;
						assigned = 1;
						break;
					}
				}
				if(!assigned)
				{
					peer_id = world_size-1;
				}

				//if(peer_id == i) continue;

				//printf("colind for process %d : %d is to be recieved from peer %d values\n", i, cur_colind, peer_id);
				mark_colind[cur_colind] = 1;

				receive_send_values[i][peer_id][receive_counters[i][peer_id]] = cur_colind;
				send_receive_values[peer_id][i][send_counters[peer_id][i]] = cur_colind;
				receive_counters[i][peer_id]++;
				send_counters[peer_id][i]++;
			}
		}
	}	



	/*for(int i=0;i<world_size;i++)
	{
		for(int j=0;j<world_size;j++)
		{
			//printf("Receive to send values %d to %d is %d, values are :\n", i,j, receive_send_counts[i][j]);
			//printIntList(receive_send_values[i][j], receive_send_counts[i][j]);

			//printf("Send to receive values %d to %d is %d, values are :\n", i,j, send_receive_counts[i][j]);
			//printIntList(send_receive_values[i][j], send_receive_counts[i][j]);

		}
	}*/


		int *g_counts = recv_counts;
		int *g_displs = recv_displs;

		////printIntList(recv_counts,world_size);
		////printIntList(recv_displs,world_size);

		for(int i=0; i<world_size;i++)
		{
			MPI_Request req8;
			MPI_Isend(recv_counts, world_size, MPI_INT, i, RECV_COUNTS, MPI_COMM_WORLD, &req8);
			if(i==0)
			{
				recv_c = (int*) malloc(sizeof(int)*world_size);
				MPI_Recv(recv_c, world_size, MPI_INT, 0, RECV_COUNTS, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

			}

			MPI_Request req9;
			MPI_Isend(recv_displs, world_size, MPI_INT, i, RECV_DISPLS, MPI_COMM_WORLD, &req9);
			if(i==0)
			{
				recv_d = (int*) malloc(sizeof(int)*world_size);
				MPI_Recv(recv_d, world_size, MPI_INT, 0, RECV_DISPLS, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
			}
		}




		for(int i=0; i<world_size;i++)
			{
				MPI_Request req8;
				MPI_Isend(receive_send_counts[i], world_size, MPI_INT, i, RECV_FROM_OTHERS_COUNT, MPI_COMM_WORLD, &req8);

				if(i==0)
				{
					to_receive_counts = (int*) malloc(sizeof(int)*world_size);
					MPI_Recv(to_receive_counts, world_size, MPI_INT, 0, RECV_FROM_OTHERS_COUNT, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
				}

				MPI_Request req9;
				MPI_Isend(send_receive_counts[i], world_size, MPI_INT, i, SEND_TO_OTHERS_COUNT, MPI_COMM_WORLD, &req9);
				if(i==0)
				{
					to_send_counts = (int*) malloc(sizeof(int)*world_size);
					MPI_Recv(to_send_counts, world_size, MPI_INT, 0, SEND_TO_OTHERS_COUNT, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
				}


				MPI_Request req11;
				for(int tr=0; tr<world_size;tr++)
				{
					if(receive_send_counts[i][tr]>0)
					{
						MPI_Isend(receive_send_values[i][tr], receive_send_counts[i][tr], MPI_INT, i, RECV_FROM_OTHERS_INDICES, MPI_COMM_WORLD, &req11);
					}
				}
				if(i==0)
				{
					for(int tr=0; tr<world_size;tr++)
					{
						if(to_receive_counts[tr]>0)
						{
							to_receive_values[tr] = (int *) malloc(sizeof(int)*to_receive_counts[tr]);
							MPI_Recv(to_receive_values[tr], to_receive_counts[tr], MPI_INT, 0, RECV_FROM_OTHERS_INDICES, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
						}
					}
				}


				MPI_Request req10;
				for(int tr=0; tr<world_size;tr++)
				{
					if(send_receive_counts[i][tr]>0)
					{
						MPI_Isend(send_receive_values[i][tr], send_receive_counts[i][tr], MPI_INT, i, SEND_TO_OTHERS_INDICES, MPI_COMM_WORLD, &req10);
					}
				}

				if(i==0)
				{
					for(int ts=0; ts<world_size;ts++)
					{
						if(to_send_counts[ts]>0)
						{
							to_send_values[ts] = (int *) malloc(sizeof(int)*to_send_counts[ts]);
							MPI_Recv(to_send_values[ts], to_send_counts[ts], MPI_INT, 0, SEND_TO_OTHERS_INDICES, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
						}
					}
				}
			}




		//Process 0 is done with its broker like duties at this point.
		//Now we do the pagerank logic for it

		//Local prep before page rank

		double alpha = 0.85;
		double * e = (double*) malloc(sizeof(double)* n_all);
		for(int i=0;i<n_all;i++)
		{
			e[i] = 1;
		}

		double *r;
		int iterations =0;
		
		//Check if all processes received all prerequisites at this point
		MPI_Barrier(MPI_COMM_WORLD);

		//printf("Starting Page rank===========\n");
		//printDoubleList(prank, n_all);


		do
		{
			//printf("---------------------------------------------------Iteration %d---------------------------------------------\n",iterations);
					//Send out the partial page rank vector to each process
					int extra_work = n_all % world_size;
					int mod = extra_work;
					for(int i=0; i<world_size ;i++)
					{
						int start_index;
						int end_index;
						if(extra_work!=0)
						{
							////printf("Extra work %d\n", extra_work);
							start_index = i*(rows_per_process+1);
							end_index =(i+1)*(rows_per_process+1);
							extra_work--;	
						}
						else
						{
							start_index = i*(rows_per_process) + mod;
							end_index =(i+1)*(rows_per_process) + mod;
						}
							MPI_Request req10;
							int n_part = (end_index - start_index);
							//double* part_prank = (double*) malloc(sizeof(double)*(n_part));
							//memcpy(part_prank, prank + start_index , sizeof(double) * (n_part));
							////printIntList(part_rbegin, n_part+1);
							//printf("%d Part of pagerank\n", n_part);
							//printDoubleList(prank +start_index, n_part);
							//printf("End of Part of pagerank\n");
							MPI_Isend(prank + start_index, n_part, MPI_DOUBLE, i, PRANK, MPI_COMM_WORLD, &req10);
							//free(part_prank);
							if(i==0)
							{

								prank_part = (double*) malloc(sizeof(double)*n);
								MPI_Recv(prank_part, n, MPI_DOUBLE, 0, PRANK, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
								//printf("Process %d received prank \n", world_rank);
								//printDoubleList(prank_part, n);
							}
					}


					/*gettimeofday(&tz, &tx);
			      	end_time = (double)tz.tv_sec + (double) tz.tv_usec / 1000000.0;
			        time_elapsed = (end_time - start_time);
			   		//printDoubleList(prank,n_all);
			        printf("Time elapsed  before send buf %lf\n", time_elapsed);*/

					prank_full = (double*) malloc (sizeof(double)*n_a);
					for(int i=0; i<world_size;i++)
					{
						if(to_send_counts[i]>0)
						{
							if(i!=world_rank)
							{
									//put required values in a vector.
								double * send = (double *) malloc(sizeof(double)*to_send_counts[i]);
								for(int j=0;j<to_send_counts[i];j++)
								{
									send[j] = prank_part[to_send_values[i][j] - x_offset];
								}
								//printf("Sending from process %d to process %d partial values:\n",world_rank, i);
								//printDoubleList(send, to_send_counts[i] );
								MPI_Request req;
								MPI_Isend(send, to_send_counts[i], MPI_DOUBLE, i, X_PART, MPI_COMM_WORLD, &req);
							}
						}
		}

		for(int i=0; i<world_size;i++)
		{
			if(to_receive_counts[i]>0)
			{
							if(i!=world_rank)
							{
				received_values[i] = (double *) malloc(sizeof(double) * to_receive_counts[i]);
				MPI_Recv(received_values[i], to_receive_counts[i], MPI_DOUBLE, i, X_PART, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
				//printf("Received by process %d from process %d partial values:\n",world_rank, i);
				//printDoubleList(received_values[i], to_receive_counts[i]);
				for(int j=0;j<to_receive_counts[i];j++)
				{
					prank_full[to_receive_values[i][j]] = received_values[i][j];
				}
				//free(received_values[i]);
			}
			else
			{
				for(int j=0;j<to_receive_counts[i];j++)
				{
					prank_full[to_receive_values[i][j]] = prank_part[to_receive_values[i][j] - x_offset];
				}
			}
			}
		}
					/*gettimeofday(&tz, &tx);
			      	end_time = (double)tz.tv_sec + (double) tz.tv_usec / 1000000.0;
			        time_elapsed = (end_time - start_time);
			   		//printDoubleList(prank,n_all);
			        printf("Time elapsed after send buf %lf\n", time_elapsed);*/

					//update the time elapsed (time-step)
			      	/*gettimeofday(&tz, &tx);
			      	end_time = (double)tz.tv_sec + (double) tz.tv_usec / 1000000.0;
			        time_elapsed = (end_time - start_time);
			   		//printDoubleList(prank,n_all);
			        printf("Time elapsed is before all to all at %d  %lf\n", world_rank, time_elapsed);*/

					//free(prank_send_buffer);

					//printf("Process %d list\n",world_rank);
					//printDoubleList(prank_full,n_a);

					//update the time elapsed (time-step)
			      	/*gettimeofday(&tz, &tx);
			      	end_time = (double)tz.tv_sec + (double) tz.tv_usec / 1000000.0;
			        time_elapsed = (end_time - start_time);
			   		//printDoubleList(prank,n_all);
			        printf("Time elapsed is after all to all at %d %lf\n", world_rank, time_elapsed);*/

					double* y;

					/*gettimeofday(&tz, &tx);
			      	end_time = (double)tz.tv_sec + (double) tz.tv_usec / 1000000.0;
			        time_elapsed = (end_time - start_time);
			   		//printDoubleList(prank,n_all);
			        printf("Time elapsed  before matvec at 0 %lf\n", time_elapsed);*/
					CSRMatVec(values, colind, rbegin, prank_full, n, &y, offset);
					/*gettimeofday(&tz, &tx);
			      	end_time = (double)tz.tv_sec + (double) tz.tv_usec / 1000000.0;
			        time_elapsed = (end_time - start_time);
			   		//printDoubleList(prank,n_all);
			        printf("Time elapsed  before matvec at 0  %lf\n", time_elapsed);*/
					//printf("Product vector y %d list\n",world_rank);
					//printDoubleList(y, n);



					gathered_result = (double *)malloc(sizeof(double)*n_a);
					//printf("Size of gathered result %ld", sizeof(gathered_result));


					//printf("Page rank gather stored in %d doubles long vector:\n", n_a);
					//printf("Process %d contributes %d to global gather\n", world_rank, n);

					//update the time elapsed (time-step)
			      	/*gettimeofday(&tz, &tx);
			      	end_time = (double)tz.tv_sec + (double) tz.tv_usec / 1000000.0;
			        time_elapsed = (end_time - start_time);
			   		//printDoubleList(prank,n_all);
			        printf("Time elapsed is before gather %lf\n", time_elapsed);*/

					MPI_Gatherv(y, n, MPI_DOUBLE,
			                gathered_result, g_counts, g_displs,
			                MPI_DOUBLE, 0, MPI_COMM_WORLD);
					//free(y);

					//update the time elapsed (time-step)
			      	/*gettimeofday(&tz, &tx);
			      	end_time = (double)tz.tv_sec + (double) tz.tv_usec / 1000000.0;
			        time_elapsed = (end_time - start_time);
			   		//printDoubleList(prank,n_all);
			        printf("Time elapsed is after gather %lf\n", time_elapsed);*/

					//printf("Page rank vector after one it gather at 0:\n");
					//printDoubleList(gathered_result, n_a);

					prank_previous = prank;
					//printf("Previous page rank vector:\n");
					//printDoubleList(prank, n_all);

					double *y_pr;
					vector_const_multiply(&y_pr, alpha, gathered_result, n_all);
					double gamma = 1 - vector_sum(y_pr,n_all);
					double *add_to_ypr;
					vector_const_multiply(&add_to_ypr, gamma/n_all, e, n_all);
					vector_add(&prank, y_pr, add_to_ypr, n_all);
					//free(y_pr);
					//free(add_to_ypr);

					//printf("New page rank vector:\n");
					//printDoubleList(prank, n_all);

					vector_sub(&r, prank, prank_previous, n_all);

					//printf("Infinity norm%lf\n", infinity_norm(r,n_all));

					if(!(infinity_norm(r,n_all) < CONVERGENCE_LIMIT))
					{
						for(int i=1;i<world_size;i++)
						{
							int exit =0;
							MPI_Send(&exit, 1, MPI_INT, i, EXIT, MPI_COMM_WORLD);
						}
					}
			iterations++;
		}
		while(!(infinity_norm(r,n_all) < CONVERGENCE_LIMIT));

		for(int i=1;i<world_size;i++)
		{
			int exit =1;
			MPI_Send(&exit, 1, MPI_INT, i, EXIT, MPI_COMM_WORLD);
		}
		//MPI_Gather(y, n, MPI_DOUBLE, gathered_result, n_a , MPI_DOUBLE, 0, MPI_COMM_WORLD);

		//update the time elapsed (time-step)
      	gettimeofday(&tz, &tx);
      	end_time = (double)tz.tv_sec + (double) tz.tv_usec / 1000000.0;
        time_elapsed = (end_time - start_time);
   		printDoubleList(prank,n_all);
        //printf("No of iterations to converge %d,  Time elapsed is %lf\n", iterations, time_elapsed);
		MPI_Finalize();



		/*do
		{
			//Get world size and divide the the number of rows in the matrix = len(rbegin-1) by the world_size.
			//get the value of len(rbegin) % world size
			int rows_per_process = n/world_size;
			int extra_work = n%world_size;
			for(int i=0; i<world_size ;i++)
			{
				if(extra_work)
				{
					int start_index = i*(rows_per_process+1);
					int end_index =(i+1)*(rows_per_process+1);
					extra_work--;	
				}
				else
				{
					int start_index = i*(rows_per_process);
					int end_index =(i+1)*(rows_per_process);

				}

				//Send out the partial page rank vector to each process
				double* part_pr = (double *) malloc(sizeof(double)*(end_index - start_index))
				memcpy(part_pr, prank, end_index - start_index);

				

				


				


				

			}


		}
		while(convergence)*/
	}
	else
	{
		//You're a slave process do the math and return the result

		MPI_Recv(&n_a, 1, MPI_INT, 0, N_ALL, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
		//printf("Process %d received n all %d", world_rank, n_a);
		gettimeofday(&tz, &tx);
   		start_time = (double)tz.tv_sec + (double) tz.tv_usec / 1000000.0;	

		MPI_Recv(&n, 1, MPI_INT, 0, N_PART, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
		//printf("Process %d received n ", world_rank);

		MPI_Recv(&nzeros, 1, MPI_INT, 0, N_ZEROS, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
		//printf("Process %d received nzeros ", world_rank);
		
		values = (double*) malloc(sizeof(double)*nzeros);
		MPI_Recv(values, nzeros, MPI_DOUBLE, 0, VALUES, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
		//printf("Process %d received values ", world_rank);
		//printDoubleList(values, nzeros);
		
		rbegin = (int*) malloc(sizeof(int)*n+1);
		MPI_Recv(rbegin, n+1, MPI_INT, 0, RBEGIN, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
		//printf("Process %d received rbegin ", world_rank);
		//printIntList(rbegin, n+1);

		colind = (int*) malloc(sizeof(int)*nzeros);
		MPI_Recv(colind, nzeros, MPI_INT, 0, COLIND, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
		//printf("Process %d received colind ", world_rank);
		//printIntList(colind, nzeros);


		MPI_Recv(&offset, 1, MPI_INT, 0, OFFSET, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

		MPI_Recv(&x_offset, 1, MPI_INT, 0, X_OFFSET, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
		//printf("\nProcess %d received offset %d\n", world_rank, offset);

		//printf("\n All to all send count at %d is %d and recv count is %d\n", world_rank, n, n_a);

		recv_c = (int*) malloc(sizeof(int)*world_size);
		MPI_Recv(recv_c, world_size, MPI_INT, 0, RECV_COUNTS, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

		recv_d = (int*) malloc(sizeof(int)*world_size);
		MPI_Recv(recv_d, world_size, MPI_INT, 0, RECV_DISPLS, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

		to_receive_counts = (int*) malloc(sizeof(int)*world_size);
		MPI_Recv(to_receive_counts, world_size, MPI_INT, 0, RECV_FROM_OTHERS_COUNT, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

		to_send_counts = (int*) malloc(sizeof(int)*world_size);
		MPI_Recv(to_send_counts, world_size, MPI_INT, 0, SEND_TO_OTHERS_COUNT, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

		for(int tr=0; tr<world_size;tr++)
		{
			if(to_receive_counts[tr]>0)
			{
				to_receive_values[tr] = (int *) malloc(sizeof(int)*to_receive_counts[tr]);
				MPI_Recv(to_receive_values[tr], to_receive_counts[tr], MPI_INT, 0, RECV_FROM_OTHERS_INDICES, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
				//printIntList(to_receive_values[tr], to_receive_counts[tr]);
			}
		}

		for(int ts=0; ts<world_size;ts++)
		{
			if(to_send_counts[ts]>0)
			{
				to_send_values[ts] = (int *) malloc(sizeof(int)*to_send_counts[ts]);
				MPI_Recv(to_send_values[ts], to_send_counts[ts], MPI_INT, 0, SEND_TO_OTHERS_INDICES, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
				//printIntList(to_send_values[ts], to_send_counts[ts]);
			}
		}

		MPI_Barrier(MPI_COMM_WORLD);

		do
		{
		prank_part = (double*) malloc(sizeof(double)*n);
		MPI_Recv(prank_part, n, MPI_DOUBLE, 0, PRANK, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
		//printf("Process %d received prank \n ", world_rank);
		//printDoubleList(prank_part, n);

		prank_full = (double*) malloc (sizeof(double)*n_a);

		/*for(int i=0; i<world_size;i++)
		{
			if(to_send_counts[i]>0)
			{
				//put required values in a vector.
				double * send = (double *) malloc(sizeof(double)*to_send_counts[i]);
				for(int j=0;j<to_send_counts[i];j++)
				{
					send[j] = prank_part[to_send_values[i][j] - x_offset];
				}
				printf("Sending from process %d to process %d partial values:\n",world_rank, i);
				printDoubleList(send, to_send_counts[i]);
				MPI_Request req;
				MPI_Isend(send, to_send_counts[i], MPI_DOUBLE, i, X_PART, MPI_COMM_WORLD, &req);
			}
		}

		for(int i=0; i<world_size;i++)
		{
			if(to_receive_counts[i]>0)
			{
				received_values[i] = (double *) malloc(sizeof(double) * to_receive_counts[i]);
				MPI_Recv(received_values[i], to_receive_counts[i], MPI_DOUBLE, i, X_PART, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

				printf("Received by process %d from process %d partial values:\n",world_rank, i);
				printDoubleList(received_values[i], to_receive_counts[i]);

				for(int j=0;j<to_receive_counts[i];j++)
				{
					prank_full[to_receive_values[i][j]] = received_values[i][j];
				}
				//free(received_values[i]);
			}
		}*/

							for(int i=0; i<world_size;i++)
					{
						if(to_send_counts[i]>0)
						{
							if(i!=world_rank)
							{
									//put required values in a vector.
								double * send = (double *) malloc(sizeof(double)*to_send_counts[i]);
								for(int j=0;j<to_send_counts[i];j++)
								{
									send[j] = prank_part[to_send_values[i][j] - x_offset];
								}
								//printf("Sending from process %d to process %d partial values:\n",world_rank, i);
								//printDoubleList(send, to_send_counts[i] );
								MPI_Request req;
								MPI_Isend(send, to_send_counts[i], MPI_DOUBLE, i, X_PART, MPI_COMM_WORLD, &req);
							}
						}
		}

		for(int i=0; i<world_size;i++)
		{
			if(to_receive_counts[i]>0)
			{
							if(i!=world_rank)
							{
				received_values[i] = (double *) malloc(sizeof(double) * to_receive_counts[i]);
				MPI_Recv(received_values[i], to_receive_counts[i], MPI_DOUBLE, i, X_PART, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
				//printf("Received by process %d from process %d partial values:\n",world_rank, i);
				//printDoubleList(received_values[i], to_receive_counts[i]);
				for(int j=0;j<to_receive_counts[i];j++)
				{
					prank_full[to_receive_values[i][j]] = received_values[i][j];
				}
				//free(received_values[i]);
			}
			else
			{
				for(int j=0;j<to_receive_counts[i];j++)
				{
					prank_full[to_receive_values[i][j]] = prank_part[to_receive_values[i][j] - x_offset];
				}
			}
			}
		}
		

					/*gettimeofday(&tz, &tx);
			      	end_time = (double)tz.tv_sec + (double) tz.tv_usec / 1000000.0;
			        time_elapsed = (end_time - start_time);
			   		//printDoubleList(prank,n_all);
			        printf("Time elapsed is before all to all at %d %lf\n", world_rank, time_elapsed);*/
					      /*	gettimeofday(&tz, &tx);
			      	end_time = (double)tz.tv_sec + (double) tz.tv_usec / 1000000.0;
			        time_elapsed = (end_time - start_time);
			   		//printDoubleList(prank,n_all);
			        printf("Time elapsed is after all to all at %d %lf\n", world_rank, time_elapsed);*/
		//free(prank_send_buffer);

		//printf("Process %d list\n",world_rank);
		//printDoubleList(prank_full,n_a);

					double* y;
					/*gettimeofday(&tz, &tx);
			      	end_time = (double)tz.tv_sec + (double) tz.tv_usec / 1000000.0;
			        time_elapsed = (end_time - start_time);
			   		//printDoubleList(prank,n_all);
			        printf("Time elapsed  before matvec at process %d %lf\n", world_rank, time_elapsed);*/

		CSRMatVec(values, colind, rbegin, prank_full, n, &y, offset);

		/*gettimeofday(&tz, &tx);
			      	end_time = (double)tz.tv_sec + (double) tz.tv_usec / 1000000.0;
			        time_elapsed = (end_time - start_time);
			   		//printDoubleList(prank,n_all);
			        printf("Time elapsed  before matvec at process %d %lf\n", world_rank, time_elapsed);*/
		//printf("Product vector y %d list\n",world_rank);
		//printDoubleList(y, n);

		//printf("Process %d contributes %d to global gather\n", world_rank, n);
					      	/*gettimeofday(&tz, &tx);
			      	end_time = (double)tz.tv_sec + (double) tz.tv_usec / 1000000.0;
			        time_elapsed = (end_time - start_time);
			   		//printDoubleList(prank,n_all);
			        printf("Time elapsed  at %d  is before gather %lf\n", world_rank,  time_elapsed);*/
		MPI_Gatherv(y, n, MPI_DOUBLE, NULL, NULL, NULL, MPI_DOUBLE, 0, MPI_COMM_WORLD);

					      	/*gettimeofday(&tz, &tx);
			      	end_time = (double)tz.tv_sec + (double) tz.tv_usec / 1000000.0;
			        time_elapsed = (end_time - start_time);
			   		//printDoubleList(prank,n_all);
			        printf("Time elapsed at %d is before gather %lf\n", world_rank, time_elapsed);*/
		//free(y);
			        /*gettimeofday(&tz, &tx);
			      	end_time = (double)tz.tv_sec + (double) tz.tv_usec / 1000000.0;
			        time_elapsed = (end_time - start_time);
			   		//printDoubleList(prank,n_all);
			        printf("Before t %d  %lf\n", world_rank,  time_elapsed);*/	
		MPI_Recv(&exit, 1, MPI_INT, 0, EXIT, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

		/*gettimeofday(&tz, &tx);
			      	end_time = (double)tz.tv_sec + (double) tz.tv_usec / 1000000.0;
			        time_elapsed = (end_time - start_time);
			   		//printDoubleList(prank,n_all);
			        printf("After Time elapsed  at %d  is before gather %lf\n", world_rank,  time_elapsed);*/

		}
		while(!exit);
		//MPI_Gatherv(y, n, MPI_DOUBLE, gathered_result, n_a , MPI_DOUBLE, 0, MPI_COMM_WORLD);
		MPI_Finalize();
	}


	return 0;
}


void initializeRanks(double **answer, int n)
{
	(*answer) = (double*) malloc(sizeof(double)*n);
	for(int i=0;i<n;i++)
	{
		(*answer)[i] = 1.0/n;
	}
}

void printIntList(int * list, int n)
{
	for(int i=0;i<n;i++)
	{
		printf("%d ", list[i]);
	}
	printf("\n");
}

void printDoubleList(double * list, int n)
{
	for(int i=0;i<n;i++)
	{
		printf("%.16lf ",list[i]);
	}
	printf("\n");
}

//A function to perform matrix vector multiplication serially
void CSRMatVec(double * value, int * colind, int *rbegin, double * x, int n, double **y, int offset)
{
	(*y) = (double*) malloc(sizeof(double)*n); 
	for(int i=0;i<n;i++)
	{
		//printf("Pass of %d \n", i);
		//printf("Offset %d\n", offset);
		(*y)[i] = 0;
		int k1 = rbegin[i] - offset;
		int k2 = rbegin[i+1] - offset - 1;
		//printf("From  %d \n", k1);
		//printf("From %d \n", k2);
		if(k2<k1) continue;

		for(int k=k1 ;k<=k2; k++)
		{
			//printf("adding %lf x %lf to y of %d\n", value[k], x[colind[k]], i);
			(*y)[i] += value[k] * x[colind[k]];
		}
	}
}

void vector_sub(double ** result, double* vec_a, double* vec_b, int l)
{
	(*result) = (double*) malloc(sizeof(double)*l);
	for(int i=0;i<l;i++)
	{
		(*result)[i] = vec_a[i] - vec_b[i];
	}
}

void vector_add(double ** result, double* vec_a, double* vec_b, int l)
{
	(*result) = (double*) malloc(sizeof(double)*l);
	for(int i=0;i<l;i++)
	{
		(*result)[i] = vec_a[i] + vec_b[i];
	}
}

void vector_const_multiply(double ** result, double constant, double* vec_a, int l)
{
	(*result) = (double*) malloc(sizeof(double)*l);
	for(int i=0;i<l;i++)
	{
		(*result)[i] = constant * vec_a[i] ;
	}
}

double vector_sum(double* vec_a, int l)
{
	double result =0;
	for(int i=0;i<l;i++)
	{
		result +=  vec_a[i];
	}
}

double infinity_norm(double* vec_a, int l)
{
	double max = 0;
	for(int i=0;i<l;i++)
	{
		double abs_i = fabs(vec_a[i]);
		if(abs_i > max)
		{
			max = abs_i;
		}
	}

	return max;
}