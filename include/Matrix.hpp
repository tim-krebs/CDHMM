#pragma once

#include <string>

class Matrix{
private:
	int number_rows;
	int number_columns;

	int **index_M;
	int **index_N;

public:
	Matrix();
	~Matrix();

	void Inverse(std::string type_matrix, int number_rows, std::vector<std::vector<float> >M, std::vector<std::vector<float> >N);
	void Inverse(std::string type_matrix, int number_rows, std::vector<std::vector<double> >M, std::vector<std::vector<double> >N);
	void Multiplication(int M_rows, int M_columns, int N_columns, float **M, float **N, float **O);
	void Multiplication(int M_rows, int M_columns, int N_columns, double **M, double **N, double **O);
	void Transpose(int number_rows, int number_columns, float **M, float **N);
	void Transpose(int number_rows, int number_columns, double **M, double **N);

	int LU_Decomposition(int number_rows, std::vector<std::vector<float> >M, std::vector<std::vector<float> >L, std::vector<std::vector<float> >U);
	int LU_Decomposition(int number_rows, std::vector<std::vector<double> >M, std::vector<std::vector<double> >L, std::vector<std::vector<double> >U);

	float Determinant(std::string type_matrix, int number_rows, std::vector<std::vector<float> >M);
	double Determinant(std::string type_matrix, int number_rows, std::vector<std::vector<double> >M);
};

Matrix::Matrix(){
	number_rows = 0;
	number_columns = 0;

	index_M = new int*[0];
	index_N = new int*[0];
}
Matrix::~Matrix(){
	for (int i = 0; i < number_columns; i++){
		delete[] index_N[i];
	}
	for (int i = 0; i < number_rows; i++){
		delete[] index_M[i];
	}
	delete[] index_M;
	delete[] index_N;
}

void Matrix::Inverse(std::string type_matrix, int number_rows, std::vector<std::vector<float> >M, std::vector<std::vector<float> >N){
	int m = number_rows;

	if (m == 1){
		N[0][0] = 1 / M[0][0];
	}
	else
	if (m >= 2){
		if (!type_matrix.compare("diagonal")){
			for (int i = 0; i < m; i++){
				for (int j = 0; j < m; j++){
					N[i][j] = (i == j) ? (1 / M[i][j]) : (0);
				}
			}
		}
		else
		if (!type_matrix.compare("block-diagonal")){
			int recent_index = 0;

			for (int i = 0; i < m; i++){
				for (int j = 0; j < m; j++){
					N[i][j] = 0;
				}
			}
			for (int i = 0; i < m; i++){
				if (M[recent_index][i] == 0 || i == m - 1){
					std::vector<std::vector<float> >T;

					if (i == m - 1){
						i = m;
					}

					T.resize(i - recent_index);

					for (int j = 0; j < i - recent_index; j++){
						T[j].resize(i - recent_index);
					}

					for (int j = 0; j < i - recent_index; j++){
						for (int k = 0; k < i - recent_index; k++){
							T[j][k] = M[recent_index + j][recent_index + k];
						}
					}
					Inverse("full", i - recent_index, T, T);

					for (int j = 0; j < i - recent_index; j++){
						for (int k = 0; k < i - recent_index; k++){
							N[recent_index + j][recent_index + k] = T[j][k];
						}
					}

					for (int j = 0; j < i - recent_index; j++){
						T[j].clear();
					}
					T.clear();

					recent_index = i;
				}
			}
		}
		else
		if (!type_matrix.compare("full")){
			std::vector<std::vector<float> >T(m);

			for (int i = 0; i < m; i++){
				T[i].resize(m);
			}

			for (int i = 0; i < m; i++){
				for (int j = 0; j < m; j++){
					T[i][j] = M[i][j];
				}
			}

			// make identity matrix
			for (int i = 0; i < m; i++){
				for (int j = 0; j < m; j++){
					N[i][j] = (i == j) ? (1.0f) : (0);
				}
			}

			// lower triangle elimination
			for (int k = 0; k < m - 1; k++){
				for (int i = k + 1; i < m; i++){
					float ratio = T[i][k] / T[k][k];

					for (int j = k; j < m; j++){
						T[i][j] -= T[k][j] * ratio;
					}
					for (int j = 0; j < m; j++){
						N[i][j] -= N[k][j] * ratio;
					}
				}
			}

			// make diagonal to 1.0 
			for (int i = 0; i < m; i++){
				float ratio = T[i][i];

				T[i][i] = 1.0;
				for (int j = i + 1; j < m; j++){
					T[i][j] /= ratio;
				}
				for (int j = 0; j <= i; j++){
					N[i][j] /= ratio;
				}
			}

			// upper triangle elimination
			for (int k = m - 1; k > 0; k--){
				for (int i = k - 1; i >= 0; i--){
					float ratio = T[i][k];

					T[i][k] = 0;
					for (int j = 0; j < m; j++){
						N[i][j] -= N[k][j] * ratio;
					}
				}
			}

			for (int i = 0; i < m; i++){
				T[i].clear();
			}
			T.clear();
		}
	}
}
void Matrix::Inverse(std::string type_matrix, int number_rows, std::vector<std::vector<double> >M, std::vector<std::vector<double> >N){
	int m = number_rows;

	if (m == 1){
		N[0][0] = 1 / M[0][0];
	}
	else
	if (m >= 2){
		if (type_matrix.compare("diagonal")){
			for (int i = 0; i < m; i++){
				for (int j = 0; j < m; j++){
					N[i][j] = (i == j) ? (1 / M[i][j]) : (0);
				}
			}
		}
		else
		if (!type_matrix.compare("block-diagonal")){
			int recent_index = 0;

			for (int i = 0; i < m; i++){
				for (int j = 0; j < m; j++){
					N[i][j] = 0;
				}
			}
			for (int i = 0; i < m; i++){
				if (M[recent_index][i] == 0 || i == m - 1){
					std::vector<std::vector<double> > T;

					if (i == m - 1){
						i = m;
					}

					T.resize(i - recent_index);

					for (int j = 0; j < i - recent_index; j++){
						T[j].resize(i - recent_index);
					}

					for (int j = 0; j < i - recent_index; j++){
						for (int k = 0; k < i - recent_index; k++){
							T[j][k] = M[recent_index + j][recent_index + k];
						}
					}
					Inverse("full", i - recent_index, T, T);

					for (int j = 0; j < i - recent_index; j++){
						for (int k = 0; k < i - recent_index; k++){
							N[recent_index + j][recent_index + k] = T[j][k];
						}
					}

					for (int j = 0; j < i - recent_index; j++){
						T[j].clear();
					}
					T.clear();

					recent_index = i;
				}
			}
		}
		else
		if (!type_matrix.compare("full")){
			std::vector<std::vector<double> >T(m);

			for (int i = 0; i < m; i++){
				T[i].resize(m);
			}

			for (int i = 0; i < m; i++){
				for (int j = 0; j < m; j++){
					T[i][j] = M[i][j];
				}
			}

			// make identity matrix
			for (int i = 0; i < m; i++){
				for (int j = 0; j < m; j++){
					N[i][j] = (i == j) ? (1) : (0);
				}
			}

			// lower triangle elimination
			for (int k = 0; k < m - 1; k++){
				for (int i = k + 1; i < m; i++){
					double ratio = T[i][k] / T[k][k];

					for (int j = k; j < m; j++){
						T[i][j] -= T[k][j] * ratio;
					}
					for (int j = 0; j < m; j++){
						N[i][j] -= N[k][j] * ratio;
					}
				}
			}

			// make diagonal to 1.0 
			for (int i = 0; i < m; i++){
				double ratio = T[i][i];

				T[i][i] = 1.0;
				for (int j = i + 1; j < m; j++){
					T[i][j] /= ratio;
				}
				for (int j = 0; j <= i; j++){
					N[i][j] /= ratio;
				}
			}

			// upper triangle elimination
			for (int k = m - 1; k > 0; k--){
				for (int i = k - 1; i >= 0; i--){
					double ratio = T[i][k];

					T[i][k] = 0;
					for (int j = 0; j < m; j++){
						N[i][j] -= N[k][j] * ratio;
					}
				}
			}

			for (int i = 0; i < m; i++){
				T[i].clear();
			}
			T.clear();
		}
	}
}
void Matrix::Multiplication(int M_rows, int M_columns, int N_columns, float **M, float **N, float **O){
	int m = M_rows;
	int n = N_columns;
	int o = M_columns;

	int **index_M = new int*[m];
	int **index_N = new int*[n];

	float **T = new float*[m];

	for (int i = 0; i < m; i++){
		index_M[i] = new int[2];
		T[i] = new float[n];
	}
	for (int i = 0; i < n; i++){
		index_N[i] = new int[2];
	}

	for (int i = 0; i < m; i++){
		for (int j = 0; j < o; j++){
			if (M[i][j] != 0){
				index_M[i][0] = j;
				break;
			}
		}
		for (int j = o - 1; j >= 0; j--){
			if (M[i][j] != 0){
				index_M[i][1] = j + 1;
				break;
			}
		}
	}
	for (int i = 0; i < n; i++){
		for (int j = 0; j < o; j++){
			if (N[j][i] != 0){
				index_N[i][0] = j;
				break;
			}
		}
		for (int j = o - 1; j >= 0; j--){
			if (N[j][i] != 0){
				index_N[i][1] = j + 1;
				break;
			}
		}
	}
	for (int i = 0; i < m; i++){
		for (int j = 0; j < n; j++){
			int index = (index_M[i][1] < index_N[j][1]) ? (index_M[i][1]) : (index_N[j][1]);

			float sum = 0;

			for (int k = (index_M[i][0] > index_N[j][0]) ? (index_M[i][0]) : (index_N[j][0]); k < index; k++){
				sum += M[i][k] * N[k][j];
			}
			T[i][j] = sum;
		}
	}
	for (int i = 0; i < m; i++){
		for (int j = 0; j < n; j++){
			O[i][j] = T[i][j];
		}
	}

	for (int i = 0; i < m; i++){
		delete[] index_M[i];
		delete[] T[i];
	}
	for (int i = 0; i < n; i++){
		delete[] index_N[i];
	}
	delete[] index_M;
	delete[] index_N;
	delete[] T;
}
void Matrix::Multiplication(int M_rows, int M_columns, int N_columns, double **M, double **N, double **O){
	int m = M_rows;
	int n = N_columns;
	int o = M_columns;

	int **index_M = new int*[m];
	int **index_N = new int*[n];

	double **T = new double*[m];

	for (int i = 0; i < m; i++){
		index_M[i] = new int[2];
		T[i] = new double[n];
	}
	for (int i = 0; i < n; i++){
		index_N[i] = new int[2];
	}

	for (int i = 0; i < m; i++){
		for (int j = 0; j < o; j++){
			if (M[i][j] != 0){
				index_M[i][0] = j;
				break;
			}
		}
		for (int j = o - 1; j >= 0; j--){
			if (M[i][j] != 0){
				index_M[i][1] = j + 1;
				break;
			}
		}
	}
	for (int i = 0; i < n; i++){
		for (int j = 0; j < o; j++){
			if (N[j][i] != 0){
				index_N[i][0] = j;
				break;
			}
		}
		for (int j = o - 1; j >= 0; j--){
			if (N[j][i] != 0){
				index_N[i][1] = j + 1;
				break;
			}
		}
	}
	for (int i = 0; i < m; i++){
		for (int j = 0; j < n; j++){
			int index = (index_M[i][1] < index_N[j][1]) ? (index_M[i][1]) : (index_N[j][1]);

			double sum = 0;

			for (int k = (index_M[i][0] > index_N[j][0]) ? (index_M[i][0]) : (index_N[j][0]); k < index; k++){
				sum += M[i][k] * N[k][j];
			}
			T[i][j] = sum;
		}
	}
	for (int i = 0; i < m; i++){
		for (int j = 0; j < n; j++){
			O[i][j] = T[i][j];
		}
	}

	for (int i = 0; i < m; i++){
		delete[] index_M[i];
		delete[] T[i];
	}
	for (int i = 0; i < n; i++){
		delete[] index_N[i];
	}
	delete[] index_M;
	delete[] index_N;
	delete[] T;
}
void Matrix::Transpose(int number_rows, int number_columns, float **M, float **N){
	int m = number_rows;
	int n = number_columns;

	float **T = new float*[m];

	for (int i = 0; i < m; i++){
		T[i] = new float[n];
	}

	for (int i = 0; i < m; i++){
		for (int j = 0; j < n; j++){
			T[i][j] = M[j][i];
		}
	}
	for (int i = 0; i < m; i++){
		for (int j = 0; j < n; j++){
			N[i][j] = T[i][j];
		}
	}

	for (int i = 0; i < m; i++){
		delete[] T[i];
	}
	delete[] T;
}
void Matrix::Transpose(int number_rows, int number_columns, double **M, double **N){
	int m = number_rows;
	int n = number_columns;

	double **T = new double*[m];

	for (int i = 0; i < m; i++){
		T[i] = new double[n];
	}

	for (int i = 0; i < m; i++){
		for (int j = 0; j < n; j++){
			T[i][j] = M[j][i];
		}
	}
	for (int i = 0; i < m; i++){
		for (int j = 0; j < n; j++){
			N[i][j] = T[i][j];
		}
	}

	for (int i = 0; i < m; i++){
		delete[] T[i];
	}
	delete[] T;
}

int Matrix::LU_Decomposition(int number_rows, std::vector<std::vector<float> >M, std::vector<std::vector<float> >L, std::vector<std::vector<float> >U){
	int m = number_rows;

	for (int i = 0; i < m; i++){
		L[i][i] = 1;

		for (int j = i; j < m; j++){
			float sum = 0;

			for (int k = 0; k <= i - 1; k++){
				sum += L[i][k] * U[k][j];
			}
			U[i][j] = M[i][j] - sum;
		}
		for (int j = i + 1; j < m; j++){
			float sum = 0;

			for (int k = 0; k <= i - 1; k++){
				sum += L[j][k] * U[k][i];
			}
			if (U[i][i] == 0){
				return 0;
			}
			L[j][i] = (M[j][i] - sum) / U[i][i];
		}
	}
	return 1;
}
int Matrix::LU_Decomposition(int number_rows, std::vector<std::vector<double> >M, std::vector<std::vector<double> >L, std::vector<std::vector<double> >U){
	int m = number_rows;

	for (int i = 0; i < m; i++){
		L[i][i] = 1;

		for (int j = i; j < m; j++){
			double sum = 0;

			for (int k = 0; k <= i - 1; k++){
				sum += L[i][k] * U[k][j];
			}
			U[i][j] = M[i][j] - sum;
		}
		for (int j = i + 1; j < m; j++){
			double sum = 0;

			for (int k = 0; k <= i - 1; k++){
				sum += L[j][k] * U[k][i];
			}
			if (U[i][i] == 0){
				return 0;
			}
			L[j][i] = (M[j][i] - sum) / U[i][i];
		}
	}
	return 1;
}

float Matrix::Determinant(std::string type_matrix, int number_rows, std::vector<std::vector<float> >M)
{
	int m = number_rows;

	float determinant = 1;

	if (m == 1){
		determinant = M[0][0];
	}
	else
	if (m == 2){
		determinant = M[0][0] * M[1][1] - M[0][1] * M[1][0];
	}
	else
	if (m >= 3){
		if (!type_matrix.compare("diagonal")){
			for (int i = 0; i < m; i++){
				determinant *= M[i][i];
			}
		}
		else
		if (!type_matrix.compare("block-diagonal")){
			int recent_index = 0;

			for (int i = 0; i < m; i++){
				if (M[recent_index][i] == 0 || i == m - 1){
					std::vector<std::vector<float> > T;

					if (i == m - 1){
						i = m;
					}

					T.resize(i - recent_index);

					for (int j = 0; j < i - recent_index; j++){
						T[j].resize(i - recent_index);
					}
					for (int j = 0; j < i - recent_index; j++){
						for (int k = 0; k < i - recent_index; k++){
							T[j][k] = M[recent_index + j][recent_index + k];
						}
					}
					determinant *= Determinant("full", i - recent_index, T);

					for (int j = 0; j < i - recent_index; j++){
						T[j].clear();
					}
					T.clear();

					recent_index = i;
				}
			}
		}
		else
		if (!type_matrix.compare("full")){
			std::vector<std::vector<float> >L(m);
			std::vector<std::vector<float> >U(m);

			for (int i = 0; i < m; i++){
				L[i].resize(m);
				U[i].resize(m);
			}
			if (LU_Decomposition(m, M, L, U) == 0){
				determinant = 0;
			}
			for (int i = 0; i < m; i++){
				determinant *= U[i][i];

				L[i].clear();
				U[i].clear();
			}
			L.clear();
			U.clear();
		}
	}
	return determinant;
}
double Matrix::Determinant(std::string type_matrix, int number_rows, std::vector<std::vector<double> >M)
{
	int m = number_rows;

	double determinant = 1;

	if (m == 1){
		determinant = M[0][0];
	}
	else
	if (m == 2){
		determinant = M[0][0] * M[1][1] - M[0][1] * M[1][0];
	}
	else
	if (m >= 3){
		if (!type_matrix.compare("diagonal")){
			for (int i = 0; i < m; i++){
				determinant *= M[i][i];
			}
		}
		else
		if (!type_matrix.compare("block-diagonal")){
			int recent_index = 0;

			for (int i = 0; i < m; i++){
				if (M[recent_index][i] == 0 || i == m - 1){
					std::vector<std::vector<double> > T;

					if (i == m - 1){
						i = m;
					}

					T.resize(i - recent_index);

					for (int j = 0; j < i - recent_index; j++){
						T[j].resize(i - recent_index);
					}
					for (int j = 0; j < i - recent_index; j++){
						for (int k = 0; k < i - recent_index; k++){
							T[j][k] = M[recent_index + j][recent_index + k];
						}
					}
					determinant *= Determinant("full", i - recent_index, T);

					for (int j = 0; j < i - recent_index; j++){
						T[j].clear();
					}
					T.clear();

					recent_index = i;
				}
			}
		}
		else
		if (!type_matrix.compare("full")){
			std::vector<std::vector<double> >L(m);
			std::vector<std::vector<double> >U(m);

			for (int i = 0; i < m; i++){
				L[i].resize(m);
				U[i].resize(m);
			}
			if (LU_Decomposition(m, M, L, U) == 0){
				determinant = 0;
			}
			for (int i = 0; i < m; i++){
				determinant *= U[i][i];

				L[i].clear();
				U[i].clear();
			}
			L.clear();
			U.clear();
		}
	}
	return determinant;
}