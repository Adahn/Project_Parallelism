#include <mpi.h>
#include <unistd.h>
#include <time.h>
#include <omp.h>
#include "projet.h"

/*
When ssh configured :
ssh-agent bash
ssh-add

To compile mpi programs:
mpicc -o toto toto.c

To run mpi programs:
mpirun -n 2 -hostfile hostfile --bynode ./toto

Variable environnement : export OMP_NUM_THREADS=2
Compilation : gcc -fopenmp toto.c -o toto

Récupérer variable environnement : getenv() renvoi chaine de charactères

Accelerer le code avec l'option -O3 (compilation plus lente mais optimisation du code)

Mélange MPI et OpenMP :
mpicc -fopenmp
mpirun -x OMP_NUM_THREADS=2 -x OMP_SCHEDULE="static\,64" -n 2 ./a.out
*/

#define PROF 1 // profondeur parcourue de manière non parallèle par le processus maitre
#define CTAG 1 // MPI default tag

unsigned long long int node_searched = 0;

// MPI variables globales
int my_rank = -1;
int nb_proc = -1;
unsigned short int machines[MAX_MOVES];
int chrono = 0;

int giveMeFreeMachine()
{
	int r;

	while(1)
	{
		r = rand()%(nb_proc-1)+1;
		if (machines[r]== 1) {
			machines[r] = 0;
			return r;
		}
	}

}

void slaveEvaluate(tree_t * T, result_t * result)
{
	node_searched++;

	move_t moves[MAX_MOVES];
	int n_moves;

	result->score = -MAX_SCORE - 1;
	result->pv_length = 0;

	if (test_draw_or_victory(T, result))
	return;

	//if (TRANSPOSITION_TABLE && tt_lookup(T, result)) /* la réponse est-elle déjà connue ? */
	//	return;
	compute_attack_squares(T);

	/* profondeur max atteinte ? si oui, évaluation heuristique */
	if (T->depth == 0) {
		result->score = (2 * T->side - 1) * heuristic_evaluation(T);
		return;
	}

	n_moves = generate_legal_moves(T, &moves[0]);

	/* absence de coups légaux : pat ou mat */
	if (n_moves == 0) {
		result->score = check(T) ? -MAX_SCORE : CERTAIN_DRAW;
		return;
	}

	if (ALPHA_BETA_PRUNING)
	sort_moves(T, n_moves, moves);

	if (T->height < PROF + 2) {

		for (int i = 0; i < n_moves; i++) {
			tree_t child;
			result_t child_result;

			#pragma omp task firstprivate(T) private(child, child_result) shared(result)
			{
				if (T->alpha < T->beta) {

					play_move(T, moves[i], &child);

					slaveEvaluate(&child, &child_result);

					int child_score = -child_result.score;

					if (child_score > result->score) {
						result->score = child_score;
						result->best_move = moves[i];
						result->pv_length = child_result.pv_length + 1;
						for(int j = 0; j < child_result.pv_length; j++)
						result->PV[j+1] = child_result.PV[j];
						result->PV[0] = moves[i];
					}

					T->alpha = MAX(T->alpha, child_score);

				}
			}
		}

	} else {
		for (int i = 0; i < n_moves; i++) {
			tree_t child;
			result_t child_result;

			play_move(T, moves[i], &child);

			slaveEvaluate(&child, &child_result);

			int child_score = -child_result.score;

			if (child_score > result->score) {
				result->score = child_score;
				result->best_move = moves[i];
				result->pv_length = child_result.pv_length + 1;
				for(int j = 0; j < child_result.pv_length; j++)
				result->PV[j+1] = child_result.PV[j];
				result->PV[0] = moves[i];
			}

			if (ALPHA_BETA_PRUNING && child_score >= T->beta)
			break;

			T->alpha = MAX(T->alpha, child_score);

		}
	}

}

void slaveWork()
{
	// On créé une structure MPI pour tree_t et result_t
	// Le c au début de chaque variable signifie communication (pour MPI)

	// Structure tree_t
	const int cNbrItemsTree = 14;
	int cBlockLengthTree[14] =
	{128, 128, 1,
		1, 1, 1, 1, 1,
		2, 2, 128,
		1, 1, 128};
		MPI_Datatype cTypesTree[14] =
		{MPI_CHAR, MPI_CHAR, MPI_INT,
			MPI_INT, MPI_INT, MPI_INT, MPI_INT, MPI_INT,
			MPI_INT, MPI_INT, MPI_CHAR,
			MPI_INT, MPI_UNSIGNED_LONG_LONG, MPI_UNSIGNED_LONG_LONG};
			MPI_Datatype mpi_tree_t;
			MPI_Aint cOffsetsTree[14];
			cOffsetsTree[0] = offsetof(tree_t, pieces);
			cOffsetsTree[1] = offsetof(tree_t, colors);
			cOffsetsTree[2] = offsetof(tree_t, side);
			cOffsetsTree[3] = offsetof(tree_t, depth);
			cOffsetsTree[4] = offsetof(tree_t, height);
			cOffsetsTree[5] = offsetof(tree_t, alpha);
			cOffsetsTree[6] = offsetof(tree_t, beta);
			cOffsetsTree[7] = offsetof(tree_t, alpha_start);
			cOffsetsTree[8] = offsetof(tree_t, king);
			cOffsetsTree[9] = offsetof(tree_t, pawns);
			cOffsetsTree[10] = offsetof(tree_t, attack);
			cOffsetsTree[11] = offsetof(tree_t, suggested_move);
			cOffsetsTree[12] = offsetof(tree_t, hash);
			cOffsetsTree[13] = offsetof(tree_t, history);
			MPI_Type_create_struct(cNbrItemsTree, cBlockLengthTree, cOffsetsTree, cTypesTree, &mpi_tree_t);
			MPI_Type_commit(&mpi_tree_t);

			// Structure result_t
			const int cNbrItemsResult = 4;
			int cBlockLengthResult[4] = {1, 1, 1, 128};
			MPI_Datatype cTypesResult[4] = {MPI_INT, MPI_INT, MPI_INT, MPI_INT};
			MPI_Datatype mpi_result_t;
			MPI_Aint cOffsetsResult[4];
			cOffsetsResult[0] = offsetof(result_t, score);
			cOffsetsResult[1] = offsetof(result_t, best_move);
			cOffsetsResult[2] = offsetof(result_t, pv_length);
			cOffsetsResult[3] = offsetof(result_t, PV);
			MPI_Type_create_struct(cNbrItemsResult, cBlockLengthResult, cOffsetsResult, cTypesResult, &mpi_result_t);
			MPI_Type_commit(&mpi_result_t);

			while(1)
			{
				MPI_Status status;
				const int cSource = 0;
				tree_t cChildReceive;
				result_t cChildResult;

				MPI_Recv(&cChildReceive, 1, mpi_tree_t, cSource, MPI_ANY_TAG, MPI_COMM_WORLD, &status);
				if (status.MPI_TAG == 2)
				return;
				clock_t start = clock(), diff;
				slaveEvaluate(&cChildReceive, &cChildResult);
				diff = clock() - start;
				int msec = diff * 1000 / CLOCKS_PER_SEC;
				chrono += msec;
				MPI_Send(&cChildResult, 1, mpi_result_t, cSource, CTAG, MPI_COMM_WORLD);
			}
			MPI_Type_free(&mpi_tree_t);
			MPI_Type_free(&mpi_result_t);
		}

		void evaluate(tree_t * T, result_t *result)
		{
			node_searched++;

			move_t moves[MAX_MOVES];
			int n_moves;

			result->score = -MAX_SCORE - 1;
			result->pv_length = 0;

			if (test_draw_or_victory(T, result))
			return;

			if (TRANSPOSITION_TABLE && tt_lookup(T, result)) /* la réponse est-elle déjà connue ? */
			return;

			compute_attack_squares(T);

			/* profondeur max atteinte ? si oui, évaluation heuristique */
			if (T->depth == 0) {
				result->score = (2 * T->side - 1) * heuristic_evaluation(T);
				return;
			}

			n_moves = generate_legal_moves(T, &moves[0]);

			/* absence de coups légaux : pat ou mat */
			if (n_moves == 0) {
				result->score = check(T) ? -MAX_SCORE : CERTAIN_DRAW;
				return;
			}

			if (ALPHA_BETA_PRUNING)
			sort_moves(T, n_moves, moves);

			if (T->height < PROF) {
				int ab = 0;
				for (int i = 0; i < n_moves; i++) {
					if (ab > 0) {
						break;
					}
					tree_t child;
					result_t child_result;

					#pragma omp task firstprivate(T) private(child, child_result) shared(result)
					{
						play_move(T, moves[i], &child);

						evaluate(&child, &child_result);

						int child_score = -child_result.score;

						if (child_score > result->score) {
							result->score = child_score;
							result->best_move = moves[i];
							result->pv_length = child_result.pv_length + 1;
							for(int j = 0; j < child_result.pv_length; j++)
							result->PV[j+1] = child_result.PV[j];
							result->PV[0] = moves[i];
						}

						if (ALPHA_BETA_PRUNING && child_score >= T->beta)
						ab = 1;
						//break;

						T->alpha = MAX(T->alpha, child_score);
					}
					#pragma omp taskwait

				}
			} else {
				if (T->height == PROF) {
					// On créé une structure MPI pour tree_t et result_t
					// Le c au début de chaque variable signifie communication (pour MPI)

					// Structure tree_t
					const int cNbrItemsTree = 14;
					int cBlockLengthTree[14] =
					{128, 128, 1,
						1, 1, 1, 1, 1,
						2, 2, 128,
						1, 1, 128};
						MPI_Datatype cTypesTree[14] =
						{MPI_CHAR, MPI_CHAR, MPI_INT,
							MPI_INT, MPI_INT, MPI_INT, MPI_INT, MPI_INT,
							MPI_INT, MPI_INT, MPI_CHAR,
							MPI_INT, MPI_UNSIGNED_LONG_LONG, MPI_UNSIGNED_LONG_LONG};
							MPI_Datatype mpi_tree_t;
							MPI_Aint cOffsetsTree[14];
							cOffsetsTree[0] = offsetof(tree_t, pieces);
							cOffsetsTree[1] = offsetof(tree_t, colors);
							cOffsetsTree[2] = offsetof(tree_t, side);
							cOffsetsTree[3] = offsetof(tree_t, depth);
							cOffsetsTree[4] = offsetof(tree_t, height);
							cOffsetsTree[5] = offsetof(tree_t, alpha);
							cOffsetsTree[6] = offsetof(tree_t, beta);
							cOffsetsTree[7] = offsetof(tree_t, alpha_start);
							cOffsetsTree[8] = offsetof(tree_t, king);
							cOffsetsTree[9] = offsetof(tree_t, pawns);
							cOffsetsTree[10] = offsetof(tree_t, attack);
							cOffsetsTree[11] = offsetof(tree_t, suggested_move);
							cOffsetsTree[12] = offsetof(tree_t, hash);
							cOffsetsTree[13] = offsetof(tree_t, history);
							MPI_Type_create_struct(cNbrItemsTree, cBlockLengthTree, cOffsetsTree, cTypesTree, &mpi_tree_t);
							MPI_Type_commit(&mpi_tree_t);

							// Structure result_t
							const int cNbrItemsResult = 4;
							int cBlockLengthResult[4] = {1, 1, 1, 128};
							MPI_Datatype cTypesResult[4] = {MPI_INT, MPI_INT, MPI_INT, MPI_INT};
							MPI_Datatype mpi_result_t;
							MPI_Aint cOffsetsResult[4];
							cOffsetsResult[0] = offsetof(result_t, score);
							cOffsetsResult[1] = offsetof(result_t, best_move);
							cOffsetsResult[2] = offsetof(result_t, pv_length);
							cOffsetsResult[3] = offsetof(result_t, PV);
							MPI_Type_create_struct(cNbrItemsResult, cBlockLengthResult, cOffsetsResult, cTypesResult, &mpi_result_t);
							MPI_Type_commit(&mpi_result_t);


							if (my_rank == 0) {
								int ab = 0;
								//#pragma omp parallel for
								for (int i = 0; i < n_moves; i++) {
									MPI_Status status;
									tree_t cChildSend;
									result_t cChildResult;

									play_move(T, moves[i], &cChildSend);
									// appel fonction qui renvoie num machine libre
									const int cDest = giveMeFreeMachine();

									MPI_Send(&cChildSend, 1, mpi_tree_t, cDest, CTAG, MPI_COMM_WORLD);
									MPI_Recv(&cChildResult, 1, mpi_result_t, cDest, CTAG, MPI_COMM_WORLD, &status);
									machines[cDest] = 1;

									int child_score = -cChildResult.score;
									if (child_score > result->score) {
										result->score = child_score;
										result->best_move = moves[i];
										result->pv_length = cChildResult.pv_length + 1;
										for(int j = 0; j < cChildResult.pv_length; j++)
										result->PV[j+1] = cChildResult.PV[j];
										result->PV[0] = moves[i];
									}

									if (ALPHA_BETA_PRUNING && child_score >= T->beta)
									ab = 1;
									//	break;

									T->alpha = MAX(T->alpha, child_score);


								}
							}
							MPI_Type_free(&mpi_tree_t);
							MPI_Type_free(&mpi_result_t);

						}
					}

				}


				void decide(tree_t * T, result_t *result)
				{
					for (int depth = 1;; depth++) {

						T->depth = depth;
						T->height = 0;
						T->alpha_start = T->alpha = -MAX_SCORE - 1;
						T->beta = MAX_SCORE + 1;

						printf("=====================================\n");
						evaluate(T, result);

						printf("depth: %d / score: %.2f / best_move : ", T->depth, 0.01 * result->score);
						print_pv(T, result);

						if (DEFINITIVE(result->score))
						break;
					}
				}

				int main(int argc, char **argv)
				{
					int provided;
					//MPI_Init(&argc, &argv);
					MPI_Init_thread(&argc, &argv, MPI_THREAD_FUNNELED, &provided);
					MPI_Comm_size(MPI_COMM_WORLD, &nb_proc);
					MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);

					// Structure tree_t
					const int cNbrItemsTree = 14;
					int cBlockLengthTree[14] =
					{128, 128, 1,
						1, 1, 1, 1, 1,
						2, 2, 128,
						1, 1, 128};
						MPI_Datatype cTypesTree[14] =
						{MPI_CHAR, MPI_CHAR, MPI_INT,
							MPI_INT, MPI_INT, MPI_INT, MPI_INT, MPI_INT,
							MPI_INT, MPI_INT, MPI_CHAR,
							MPI_INT, MPI_UNSIGNED_LONG_LONG, MPI_UNSIGNED_LONG_LONG};
							MPI_Datatype mpi_tree_t;
							MPI_Aint cOffsetsTree[14];
							cOffsetsTree[0] = offsetof(tree_t, pieces);
							cOffsetsTree[1] = offsetof(tree_t, colors);
							cOffsetsTree[2] = offsetof(tree_t, side);
							cOffsetsTree[3] = offsetof(tree_t, depth);
							cOffsetsTree[4] = offsetof(tree_t, height);
							cOffsetsTree[5] = offsetof(tree_t, alpha);
							cOffsetsTree[6] = offsetof(tree_t, beta);
							cOffsetsTree[7] = offsetof(tree_t, alpha_start);
							cOffsetsTree[8] = offsetof(tree_t, king);
							cOffsetsTree[9] = offsetof(tree_t, pawns);
							cOffsetsTree[10] = offsetof(tree_t, attack);
							cOffsetsTree[11] = offsetof(tree_t, suggested_move);
							cOffsetsTree[12] = offsetof(tree_t, hash);
							cOffsetsTree[13] = offsetof(tree_t, history);
							MPI_Type_create_struct(cNbrItemsTree, cBlockLengthTree, cOffsetsTree, cTypesTree, &mpi_tree_t);
							MPI_Type_commit(&mpi_tree_t);


							if (my_rank > 0) {
								slaveWork();
							} else {
								tree_t root;
								result_t result;

								if (argc < 2) {
									printf("usage: %s \"4k//4K/4P w\" (or any position in FEN)\n", argv[0]);
									exit(1);
								}

								if (ALPHA_BETA_PRUNING)
								printf("Alpha-beta pruning ENABLED\n");

								if (TRANSPOSITION_TABLE) {
									printf("Transposition table ENABLED\n");
									init_tt();
								}

								// Initialisation of dynamic MPI repartition
								for (int i = 0; i < nb_proc; i++) {
									machines[i] = 1; // all machines are free during initialisation
								}

								parse_FEN(argv[1], &root);
								print_position(&root);

								decide(&root, &result);

								printf("\nDécision de la position: ");
								switch(result.score * (2*root.side - 1)) {
									case MAX_SCORE: printf("blanc gagne\n"); break;
									case CERTAIN_DRAW: printf("partie nulle\n"); break;
									case -MAX_SCORE: printf("noir gagne\n"); break;
									default: printf("BUG\n");
								}

								for (int i = 0; i < nb_proc; i++) {
									MPI_Send(&root, 1, mpi_tree_t, i, 2, MPI_COMM_WORLD);
								}
							}

							printf("End of %d\n", my_rank);
							//printf("Chrono : %d\n", chrono);
							printf("Node searched: %llu\n", node_searched);

							MPI_Type_free(&mpi_tree_t);
							MPI_Finalize();


							return 0;
						}
