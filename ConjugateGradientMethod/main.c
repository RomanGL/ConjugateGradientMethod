#include <stdio.h>
#include <stdlib.h>
#include <conio.h>

#define RAND_SEED 7

double *expectedResult;

void ProcessInitialization(double **pMatrix, double **pVector, double **pResult, int *Size);
void ParallelResultCalculation(double **pMatrix, double **pVector, double **pResult, int *Size);
void ProcessTerminations(double **pMatrix, double **pVector, double **pResult, int *Size);

void SwapPointers(double **first, double **second);
void AllocateVectors(double **CurrentApproximation, double **PreviousApproximation,
	double **CurrentGradient, double **PreviousGradient, double **CurrentDirection,
	double **PreviousDirection, double **Denom, double *Size);
void DeleteVectors(double **CurrentApproximation, double **PreviousApproximation,
	double **CurrentGradient, double **PreviousGradient, double **CurrentDirection,
	double **PreviousDirection, double **Denom);

int main(int argc, char **argv)
{
	printf("Conjugate Gradient Method for Solving Linear Systems!\n");

	double *pMatrix;
	double *pVector;
	double *pResult;
	int size;

	if (argc > 1)
		size = atoi(argv[1]);
	else
	{
		printf("Invalid number of arguments!");
		exit(EXIT_FAILURE);
	}

	ProcessInitialization(&pMatrix, &pVector, &pResult, &size);
	ParallelResultCalculation(&pMatrix, &pVector, &pResult, &size);
	ProcessTerminations(&pMatrix, &pVector, &pResult, &size);

	_getch();
	return 0;
}

void ProcessInitialization(double **pMatrix, double **pVector, double **pResult, int *Size)
{
	int matrixSize = (*Size) * (*Size);
	*pMatrix = malloc(sizeof(double) * matrixSize);
	*pVector = malloc(sizeof(double) * (*Size));
	*pResult = malloc(sizeof(double) * (*Size));

	int diag = 0;
	int counter = 1;
	for (int i = 0; i < matrixSize; i++)
	{
		if (i == diag)
		{
			(*pMatrix)[i] = *Size + counter;
			diag += *Size + 1;
			counter++;
		}
		else
			(*pMatrix)[i] = 1;
	}

	expectedResult = malloc(sizeof(double) * (*Size));
	srand(RAND_SEED);
	for (int i = 0; i < *Size; i++)
	{
		(*pResult)[i] = 0;
		expectedResult[i] = rand() % (*Size + 1);
	}

	for (int i = 0; i < *Size; i++)
	{
		double temp = 0;
		for (int j = 0; j < *Size; j++)
			temp += (*pMatrix)[i * (*Size) + j] * expectedResult[j];

		(*pVector)[i] = temp;
	}
}

void AllocateVectors(double **CurrentApproximation, double **PreviousApproximation,
	double **CurrentGradient, double **PreviousGradient, double **CurrentDirection,
	double **PreviousDirection, double **Denom, double *Size)
{
	*CurrentApproximation = malloc(*Size * sizeof(double));
	*PreviousApproximation = malloc(*Size * sizeof(double));
	*CurrentGradient = malloc(*Size * sizeof(double));
	*PreviousGradient = malloc(*Size * sizeof(double));
	*CurrentDirection = malloc(*Size * sizeof(double));
	*PreviousDirection = malloc(*Size * sizeof(double));
	*Denom = malloc(*Size * sizeof(double));
}

void DeleteVectors(double **CurrentApproximation, double **PreviousApproximation,
	double **CurrentGradient, double **PreviousGradient, double **CurrentDirection,
	double **PreviousDirection, double **Denom)
{
	free(*CurrentApproximation);
	free(*PreviousApproximation);
	free(*CurrentGradient);
	free(*PreviousGradient);
	free(*CurrentDirection);
	free(*PreviousDirection);
	free(*Denom);
}

void SwapPointers(double **first, double **second)
{
	double *temp = *first;
	*first = *second;
	*second = temp;
}

// Conjugate Gradient Method – parallel implementation
void ParallelResultCalculation(double **pMatrix, double **pVector, double **pResult, int *Size)
{
	double *CurrentApproximation, *PreviousApproximation;
	double *CurrentGradient, *PreviousGradient;
	double *CurrentDirection, *PreviousDirection;
	double *Denom;
	double Step;
	int Iter = 1, MaxIter = *Size + 1;
	float Accuracy = 0.0001f;

	AllocateVectors(&CurrentApproximation, &PreviousApproximation,
		&CurrentGradient, &PreviousGradient, &CurrentDirection, &PreviousDirection,
		&Denom, &Size);

	for (int i = 0; i < *Size; i++) {
		PreviousApproximation[i] = 0;
		PreviousDirection[i] = 0;
		PreviousGradient[i] = -(*pVector)[i];
	}

	do {
		if (Iter > 1) {
			SwapPointers(&PreviousApproximation, &CurrentApproximation);
			SwapPointers(&PreviousGradient, &CurrentGradient);
			SwapPointers(&PreviousDirection, &CurrentDirection);
		}

		//compute gradient
#pragma omp parallel for
		for (int i = 0; i < *Size; i++) {
			CurrentGradient[i] = -(*pVector)[i];
			for (int j = 0; j < *Size; j++)
				CurrentGradient[i] += (*pMatrix)[i * (*Size) + j] * PreviousApproximation[j];
		}

		//compute direction
		double IP1 = 0, IP2 = 0;
#pragma omp parallel for reduction(+:IP1,IP2)
		for (int i = 0; i < *Size; i++) {
			IP1 += CurrentGradient[i] * CurrentGradient[i];
			IP2 += PreviousGradient[i] * PreviousGradient[i];
		}
#pragma omp parallel for
		for (int i = 0; i < *Size; i++) {
			CurrentDirection[i] = -CurrentGradient[i] +
				PreviousDirection[i] * IP1 / IP2;
		}

		//compute size step
		IP1 = 0;
		IP2 = 0;
#pragma omp parallel for reduction(+:IP1,IP2)
		for (int i = 0; i < *Size; i++)
		{
			Denom[i] = 0;
			for (int j = 0; j < *Size; j++)
			{
				Denom[i] += (*pMatrix)[i * (*Size) + j] * CurrentDirection[j];
			}

			IP1 += CurrentDirection[i] * CurrentGradient[i];
			IP2 += CurrentDirection[i] * Denom[i];
		}
		Step = -IP1 / IP2;

		//compute new approximation
#pragma omp parallel for
		for (int i = 0; i < *Size; i++)
		{
			CurrentApproximation[i] = PreviousApproximation[i] +
				Step * CurrentDirection[i];
		}
		Iter++;

	} while ((Dest(&PreviousApproximation, &CurrentApproximation, Size) > Accuracy) 
		&& (Iter < MaxIter));

	for (int i = 0; i < *Size; i++)
		(*pResult)[i] = CurrentApproximation[i];

	printf("Iterations = %d\n", Iter - 1);
	DeleteVectors(CurrentApproximation, PreviousApproximation, CurrentGradient,
		PreviousGradient, CurrentDirection, PreviousDirection, Denom);
}

void ProcessTerminations(double **pMatrix, double **pVector, double **pResult, int *Size)
{
	int error = 0;
	for (int i = 0; i < *Size && !error; i++)
	{
		if ((*pResult)[i] != expectedResult[i])
		{
			printf("Solution is incorrect!");
			error = 1;
		}
	}

	free(*pMatrix);
	free(*pVector);
	free(*pResult);
	free(expectedResult);

	if (error)
		exit(EXIT_FAILURE);
	else
		printf("Correct solution!");
}

double Dest(double **first, double **second, int *Size)
{
	double result = 0;
	for (int i = 0; i < *Size; i++)
	{
		result += (*first)[i] * (*second)[i];
	}

	return result;
}
