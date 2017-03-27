#include "projet.h"
#include <mpi.h>

/* 2017-02-23 : version 1.0 */

unsigned long long int node_searched = 0;
int MPI_best_source = 0;
int my_rank;
int nb_proc;

void evaluate(tree_t * T, result_t *result)
{
	node_searched++;
	
	move_t moves[MAX_MOVES];
	int n_moves;

	result->score = -MAX_SCORE - 1;
	result->pv_length = 0;
	
	if (test_draw_or_victory(T, result))
		return;

				if (TRANSPOSITION_TABLE && tt_lookup(T, result))     /* la réponse est-elle déjà connue ? */
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

	if (T->height == 0)
	{
		//printf("n_moves = %d\n", n_moves);
		/* évalue récursivement les positions accessibles à partir d'ici */
		// Variables locales 
		//int nb_proc = 0;
		//int my_rank = 0;
		//MPI_Comm_size(MPI_COMM_WORLD, &nb_proc);
		//MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);
		MPI_Status status;
		int MPI_best_score = 0;
		//int MPI_best_source = 0;
		int temp = 0;
		int it = 0;

		for (int i = 0; i < n_moves; i++) {
			if (n_moves > nb_proc)
			{
				;//printf("ALERT CODE PAS ENCORE ADAPTE\n");
				if (i % nb_proc == my_rank)
					;//printf("n_moves > nb_proc ///// rank = %d it = %d\n", my_rank, it);
			}
			if (i % nb_proc == my_rank)
			{
				tree_t child;
				result_t child_result;
				
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
				//printf("rank = %d it = %d score = %d\n", my_rank, it, result->score);
				it++;
			}
		}
		// Echange des messages (scores) et traitement de ceux-ci
		//printf("Score : %d for source %d\n", result->score, my_rank);
		if (my_rank == 0)
		{
			//MPI_best_score = result->score;
			MPI_best_score = 0;
			for (int source = 1; source < nb_proc; source++)
			{
				MPI_Recv(&temp, 1, MPI_INT, source, 0, MPI_COMM_WORLD, &status);
				if (DEFINITIVE(temp))
				{
					MPI_best_score = temp;
					MPI_best_source = source;
				}
				//printf("Source : %d and score is : %d / best score %d\n", source, temp, MPI_best_score);
			}
			if (DEFINITIVE(result->score))
			{
				MPI_best_score = result->score;
				MPI_best_source = 0;
			}
		} else {
			MPI_Send(&result->score, 1, MPI_INT, 0, 0, MPI_COMM_WORLD);
		}
		MPI_Bcast(&MPI_best_source, 1, MPI_INT, 0, MPI_COMM_WORLD);
		//printf("Best source : %d\n", MPI_best_source);
		//printf("Best score : %d for source %d\n", MPI_best_score, MPI_best_source);
		
	} else {
		if (T->height == 1)
		{
			#pragma omp parallel
			#pragma omp for
			for (int i = 0; i < n_moves; i++) {
				tree_t child;
				result_t child_result;

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

				// break non autorisé dans la version basique du parallelisme
				//if (ALPHA_BETA_PRUNING && child_score >= T->beta)
				//	break;    

				T->alpha = MAX(T->alpha, child_score);
			}
		} else {
			/* évalue récursivement les positions accessibles à partir d'ici */
			for (int i = 0; i < n_moves; i++) {

				tree_t child;
				result_t child_result;
				
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

				//if (ALPHA_BETA_PRUNING && child_score >= T->beta)
				//	break;    

				//T->alpha = MAX(T->alpha, child_score);
			}
		}
	}
	if (TRANSPOSITION_TABLE)
		tt_store(T, result);
}


void decide(tree_t * T, result_t *result)
{
	int end = 0;
	int temp = 0;
	MPI_Status status;

	for (int depth = 1;; depth++) {
		T->depth = depth;
		T->height = 0;
		T->alpha_start = T->alpha = -MAX_SCORE - 1;
		T->beta = MAX_SCORE + 1;
		if (my_rank == MPI_best_source)
		{
			printf("=====================================\n");
		}
		
		evaluate(T, result);

		if (my_rank == MPI_best_source)
		{
			//printf("Winning rank : %d and my rank = %d\n", MPI_best_source ,my_rank);
			printf("depth: %d / score: %.2f / best_move : ", T->depth, 0.01 * result->score);
			print_pv(T, result);
		}
		end = DEFINITIVE(result->score);
		if (my_rank == 0)
		{
			for (int source = 1; source < nb_proc; source++)
			{
				MPI_Recv(&temp, 1, MPI_INT, source, 0, MPI_COMM_WORLD, &status);
				if (temp > 0)
				{
					//printf("End received : %d\n", result->score);
					end = 1;
				}
			}
		} else {
			MPI_Send(&end, 1, MPI_INT, 0, 0, MPI_COMM_WORLD);
		}
		MPI_Bcast(&end, 1, MPI_INT, 0, MPI_COMM_WORLD);
		//printf("End = %d\n", end);
		if (end)
			break;
	}
}

int main(int argc, char **argv)
{  
	tree_t root;
	result_t result;

	// MPI 
	//int my_rank;
	//int nb_proc;
	MPI_Status status;

	MPI_Init(&argc, &argv);
	MPI_Comm_size(MPI_COMM_WORLD, &nb_proc);
	MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);

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
	
	parse_FEN(argv[1], &root);
	print_position(&root);
	

    float start, end;
    MPI_Barrier(MPI_COMM_WORLD); // to get sure all children are ready
    start = MPI_Wtime();

	decide(&root, &result);

    MPI_Barrier(MPI_COMM_WORLD);
    end = MPI_Wtime();

	if (my_rank == MPI_best_source)
	{
		//printf("Score final : %d and I'm %d\n", result.score, my_rank);

		printf("Runtime = %f\n", end-start);

		printf("\nDécision de la position: ");
		switch(result.score * (2*root.side - 1)) {
			case MAX_SCORE: printf("blanc gagne\n"); break;
			case CERTAIN_DRAW: printf("partie nulle\n"); break;
			case -MAX_SCORE: printf("noir gagne\n"); break;
			default: printf("BUG\n");


		}
	}
	

	printf("Node searched: %llu\n", node_searched);
	
	if (TRANSPOSITION_TABLE)
		free_tt();

	MPI_Finalize();

	return 0;
}
