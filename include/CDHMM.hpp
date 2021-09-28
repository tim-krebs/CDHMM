#pragma once

#include <string>
#include <vector>
#include <map>


#include "GMM.hpp"


class CDHMM
{
private:
    /* data */
    int number_states;
    int number_gaussians;
    int dimension_feature;

    std::string type_covariance;
    std::string type_model;

public:
    // Constructor/Destructor
    CDHMM(int number_states, std::string type_model, std::string type_covariance, int dimension_feature, int number_gaussians);
    ~CDHMM();

    // Initital Probability pi
    std::vector<double> initial_probability;

    // Transition Probability A
    std::vector<std::vector<double> > state_transition_probability;

    // Observation Probability B
    std::vector<std::vector<double> > state_observation_probability;

    // Create Vector of mixture models for discrete Observation B
    std::vector<GMM> v_GMM;

// Inititale Wahrscheinlichkeit berechenen!!!! pi = 1 0 0 0 0 0 0...
    void Initialize(int dimension_feature, int number_gaussians, int number_data, std::vector<std::vector<double> > &data);
    double Forward_Algorithm(int length_data, std::vector<int> &state, std::vector<std::vector<double> > &alpha, std::vector<double> &likelihood);
    double Backward_Algorithm(int length_data, std::vector<int> &state, std::vector<std::vector<double> > &beta, std::vector<double> &likelihood);
    double Baum_Welch_Algorithm();
};

/**
 * @brief Construct a new CDHMM::CDHMM object
 * 
 * @param number_states     (int)       Number of states of the HMM
 * @param type_model        (string)    Type of the model. Curretnly only avalable for linear hmm
 * @param type_covariance   (string)    linear or full. Currently only diagonal available
 * @param dimension_feature (int)       Dimension of the feature vector. Default is 12
 * @param number_gaussians  (int)       observations per state
 */
CDHMM::CDHMM(int number_states, std::string type_model, std::string type_covariance, int dimension_feature, int number_gaussians)
{
    this->number_states = number_states;
    this->number_gaussians = number_gaussians;
    this->dimension_feature = dimension_feature;
    this->type_covariance = type_covariance;
    this->type_model = type_model;

    v_GMM.resize(number_states);
    for(int i = 0; i < number_states; i++) 
    {
        v_GMM[i] = GMM(type_covariance, dimension_feature, number_gaussians);
    }
}

CDHMM::~CDHMM()
{
}

/**
 * @brief 
 * 
 * @param dimension_feature 
 * @param number_gaussians 
 * @param number_data 
 * @param data 
 */
void CDHMM::Initialize(int dimension_feature, int number_gaussians, int number_data, std::vector<std::vector<double> > &data)
{
    // Initital Probability pi
    for(int i = 0; i < number_states; i++)
    {
        if(i == 0) initial_probability.push_back(1);
        initial_probability.push_back(0);
    }
    for(int i = 0; i < number_states; i++)
    {
        std::cout << initial_probability[i] << " ";
    }

    //Initialize the transition prob matrix A
    state_transition_probability.resize(number_states);
    for (int i = 0; i < number_states; i++) 
    {
        state_transition_probability[i].resize(number_states);
			for (int j = 0; j < number_states; j++) 
            {
                // for linear model
				if (i == j)
                {
                    state_transition_probability[i][j] = 0.7;
                    state_transition_probability[i][j+1] = 0.3;
                }
			}
		}
    state_transition_probability[0][0] = 0.0;
    state_transition_probability[0][1] = 1.0;
    state_transition_probability[number_states-1][number_states-1] = 0.0;



    //Initialize the observation matrix B
    // Calculate the continious observations first
    Kmeans kmeans = Kmeans(data[0].size(), number_gaussians);
    kmeans.Initialize(data.size(), data);
    while (kmeans.Cluster(data.size(), data));

    for(int i = 0; i < this->number_states; i++)
    {
        for(int j = 0; j < number_gaussians; j++)
        {
            for(int k = 0; k < data[0].size(); k++)
            {
                v_GMM[i].diagonal_covariance[j][k] = 1;

                v_GMM[i].mean[j][k] = kmeans.centroid[j][k];
            }
            v_GMM[i].weight[j] = 1.0 / number_gaussians;
        }
    }


    data.clear();
}

// Woking on 
double CDHMM::Forward_Algorithm(int length_event, std::vector<int> &state, std::vector<std::vector<double> > &alpha, std::vector<double> &likelihood)
{
    double log_likelihood = 0;

    for(int i = 0; i < length_event; i++)
    {
        double tmp = 0;

        if(i == 0)
        {
            for(int j = 0; j < state.size(); j++)
            {
                int k = state[j];
                tmp += (alpha[i][j] = (j == 0) * initial_probability[k] * likelihood[i * number_states * k]); 
            }
        }
        else
        {
            for(int j = 0; j < state.size(); j++)
            {
                double tmp = 0;

            }
        }
    }

    return -log_likelihood;
}


double CDHMM::Backward_Algorithm(int length_event, std::vector<int> &state, std::vector<std::vector<double> > &beta, std::vector<double> &likelihood)
{
    double log_likelihood = 0;

    return -log_likelihood;
}

double CDHMM::Baum_Welch_Algorithm()
{
    double likelihood =0;

    return likelihood;
}
