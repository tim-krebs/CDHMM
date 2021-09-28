#pragma once

#include <string>
#include <vector>

#include "Matrix.hpp"
#include "Kmeans.hpp"

class GMM
{
private:
    /* data */
    std::string type_covariance;
    int dimension_data;
    int number_gaussian_components;


public:

    std::vector<double> weight;
    std::vector<std::vector<double> > mean;

    std::vector<std::vector<double>  >diagonal_covariance;
    std::vector<std::vector<std::vector<double> > > covariance;

	GMM();
    GMM(std::string type_covariance, int dimension_data, int number_gaussian_components);
    ~GMM();

    void Initialize(int number_data, std::vector<std::vector<double> > data);
    int Classify(std::vector<double> data);
    double Calculate_Likelihood(std::vector<double> data);
    //double Calculate_Likelihood(std::vector<double> data, double gaussian_distribution);

    double Expectation_Maximation(int number_data, std::vector<std::vector<double> > data);
    double Gaussian_Distribution(std::vector<double> data, int component_index);
    double Gaussian_Distribution(std::vector<double> data, std::vector<double> mean, std::vector<double> diagonal_covariance);
    double Gaussian_Distribution(std::vector<double> data, std::vector<double> mean, std::vector<std::vector<double> >diagonal_covariance);
};

GMM::GMM()
{
}

GMM::GMM(std::string type_covariance, int dimension_data, int number_gaussian_components)
{
    this->type_covariance = type_covariance;
	this->dimension_data = dimension_data;
	this->number_gaussian_components = number_gaussian_components;

	
	weight.resize(number_gaussian_components);

    // Create 2d matrix of mean
    mean.resize(number_gaussian_components);
	for (int i = 0; i < number_gaussian_components; i++)
    {
		mean[i].resize(dimension_data);
	}

    // Create a diagonal 2d matrix of diagonal_covariance
	if (!type_covariance.compare("diagonal"))
    {
		diagonal_covariance.resize(number_gaussian_components);

		for (int i = 0; i < number_gaussian_components; i++)
        {
			diagonal_covariance[i].resize(dimension_data);
		}
	}
	else
    {
		covariance.resize(number_gaussian_components);

		for (int i = 0; i < number_gaussian_components; i++)
        {
			covariance[i].resize(dimension_data);

			for (int j = 0; j < dimension_data; j++)
            {
				covariance[i][j].resize(dimension_data);
			}
        }
    }
}

GMM::~GMM()
{
}

void GMM::Initialize(int number_data, std::vector<std::vector<double> > data)
{
    Kmeans kmeans = Kmeans(dimension_data, number_gaussian_components);

    // Initialize Cluster
    kmeans.Initialize(number_data, data);
	while (kmeans.Cluster(number_data, data))

    // Initialize the matrix, mean and weights
    for (int i = 0; i < number_gaussian_components; i++)
    {
		for (int j = 0; j < dimension_data; j++)
        {
			if (!type_covariance.compare("diagonal"))
            {
				diagonal_covariance[i][j] = 1;
			}
			else
            {
				for (int k = 0; k < dimension_data; k++)
                {
					covariance[i][j][k] = (j == k);
				}
			}
		}
		for (int j = 0; j < dimension_data; j++)
        {
			mean[i][j] = kmeans.centroid[i][j];
		}
		weight[i] = 1.0 / number_gaussian_components;
	}
}

double GMM::Calculate_Likelihood(std::vector<double> data)
{
	double likelihood = 0;

	for (int i = 0; i < number_gaussian_components; i++)
    {
		likelihood += weight[i] * Gaussian_Distribution(data, i);
	}
	return likelihood;
}

double GMM::Expectation_Maximation(int number_data, std::vector<std::vector<double> > data)
{
    double log_likelihood = 0;

    std::vector<double> gaussian_distribution(number_gaussian_components);
    std::vector<double> sum_likelihood(number_gaussian_components);
    std::vector<std::vector<double> >new_mean(number_gaussian_components);

    std::vector<std::vector<double> >new_diagonal_covariance;
    std::vector<std::vector<std::vector<double> > >new_covariance;

    // Initialize empty new_mean matrix and sum_likelihood
    for(int i = 0; i < number_gaussian_components; i++)
    {
        new_mean[i].resize(dimension_data);

        for(int j = 0; j < dimension_data; j++)
        {
            new_mean[i][j] = 0;
        }
        sum_likelihood[i] = 0;
    }

    // Fill empty new_diagonal_covariance with zeros
    if (!type_covariance.compare("diagonal")){
		new_diagonal_covariance.resize(number_gaussian_components);

		for (int i = 0; i < number_gaussian_components; i++){
			new_diagonal_covariance[i].resize(dimension_data);

			for (int j = 0; j < dimension_data; j++){
				new_diagonal_covariance[i][j] = 0;
			}
		}
	}
	else{
        // Fill empty new_covariance with zeros
		new_covariance.resize(number_gaussian_components);

		for (int i = 0; i < number_gaussian_components; i++){
			new_covariance[i].resize(dimension_data);

			for (int j = 0; j < dimension_data; j++){
				new_covariance[i][j].resize(dimension_data);

				for (int k = 0; k < dimension_data; k++){
					new_covariance[i][j][k] = 0;
				}
			}
		}
    }

    // 
    for(int i = 0; i < number_data; i++)
    {
        double sum = 0;

        for(int j = 0; j < number_gaussian_components; j++)
        {
            if (!type_covariance.compare("diagonal"))
            {
                // Calculate the gaussian distrubution
                sum += weight[j] * (gaussian_distribution[j] = Gaussian_Distribution(data[i], mean[j], diagonal_covariance[j]));
            }
            else
            {
                // Calculate the gaussian distrubution
                sum += weight[j] * (gaussian_distribution[j] = Gaussian_Distribution(data[i], mean[j], covariance[j]));
            }
        }

        // Calculate the likelihood over all components
        for (int j = 0; j < number_gaussian_components; j++)
        {
			double likelihood = weight[j] * gaussian_distribution[j] / sum;

			for (int k = 0; k < dimension_data; k++)
            {
				if (!type_covariance.compare("diagonal"))
                {
					new_diagonal_covariance[j][k] += likelihood * (data[i][k] - mean[j][k]) * (data[i][k] - mean[j][k]);
				}
				else
                {
					for (int l = 0; l < dimension_data; l++)
                    {
						new_covariance[j][k][l] += likelihood * (data[i][k] - mean[j][k]) * (data[i][l] - mean[j][l]);
					}
				}
				new_mean[j][k] += likelihood * data[i][k];
			}
			sum_likelihood[j] += likelihood;
        }

    }
    for (int i = 0; i < number_gaussian_components; i++)
    {
		for (int j = 0; j < dimension_data; j++)
        {
			if (!type_covariance.compare("diagonal"))
            {
				diagonal_covariance[i][j] = new_diagonal_covariance[i][j] / sum_likelihood[i];
			}
			else
            {
				for (int k = 0; k < dimension_data; k++)
                {
					covariance[i][j][k] = new_covariance[i][j][k] / sum_likelihood[i];
				}
			}
			mean[i][j] = new_mean[i][j] / sum_likelihood[i];
		}
		weight[i] = sum_likelihood[i] / number_data;
	}

    // Calculate the likelihood over all data
    for (int i = 0; i < number_data; i++)
    {
		log_likelihood += log(Calculate_Likelihood(data[i]));
	}


    // Free space
	if (!type_covariance.compare("diagonal"))
    {
		for (int i = 0; i < number_gaussian_components; i++)
        {
			new_diagonal_covariance[i].clear();
		}
		new_diagonal_covariance.clear();
	}
    else
    {
		for (int i = 0; i < number_gaussian_components; i++)
        {
			for (int j = 0; j < dimension_data; j++)
            {
				new_covariance[i][j].clear();
			}
			new_covariance[i].clear();
		}
		new_covariance.clear();
	}
    for (int i = 0; i < number_gaussian_components; i++)
    {
		new_mean[i].clear();
	}
	gaussian_distribution.clear();
	new_mean.clear();
	sum_likelihood.clear();

	return log_likelihood;

}

double GMM::Gaussian_Distribution(std::vector<double> data, int component_index)
{
    int j = component_index;

	if (!type_covariance.compare("diagonal")){
		return Gaussian_Distribution(data, mean[j], diagonal_covariance[j]);
	}
	else{
		return Gaussian_Distribution(data, mean[j], covariance[j]);
	}
}

double GMM::Gaussian_Distribution(std::vector<double> data, std::vector<double> mean, std::vector<double> diagonal_covariance){
	double determinant = 1;
	double result;
	double sum = 0;

	for (int i = 0; i < dimension_data; i++){
		determinant *= diagonal_covariance[i];
		sum += (data[i] - mean[i]) * (1 / diagonal_covariance[i]) * (data[i] - mean[i]);
	}
	result = 1.0 / (pow(2 * 3.1415926535897931, dimension_data / 2.0) * sqrt(determinant)) * exp(-0.5 * sum);

    return result;
}

double GMM::Gaussian_Distribution(std::vector<double> data, std::vector<double> mean, std::vector<std::vector<double> >covariance)
{
	double result;
	double sum = 0;

	std::vector<std::vector<double> > inversed_covariance(dimension_data);

	Matrix matrix;

	for (int i = 0; i < dimension_data; i++){
		inversed_covariance[i].resize(dimension_data);
	}
	matrix.Inverse(type_covariance, dimension_data, covariance, inversed_covariance);

	for (int i = 0; i < dimension_data; i++){
		double partial_sum = 0;

		for (int j = 0; j < dimension_data; j++){
			partial_sum += (data[j] - mean[j]) * inversed_covariance[j][i];
		}
		sum += partial_sum * (data[i] - mean[i]);
	}

	for (int i = 0; i < dimension_data; i++){
		inversed_covariance[i].clear();
	}
	inversed_covariance.clear();

	result = 1.0 / (pow(2 * 3.1415926535897931, dimension_data / 2.0) * sqrt(matrix.Determinant(type_covariance, dimension_data, covariance))) * exp(-0.5 * sum);

	return result;
}
