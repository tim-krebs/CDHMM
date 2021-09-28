#include <iostream>

#include "MFCC.hpp"
#include "GMM.hpp"
#include "CDHMM.hpp"
#include "DataHandler.hpp"
#include "DynamicArray.hpp"
#include "ProjectConfig.h"

#define NUM_WORDS   1

typedef std::chrono::high_resolution_clock Clock;
typedef std::chrono::milliseconds Milliseconds;

const double MFCC::PI  = 3.14159265358979323846;
const double MFCC::PI2 = 2*PI;
const double MFCC::PI4 = 4*PI;

int main()
{
    // Check Project Version
    std::cout << "Project Version: " << PROJECT_VERSION_MAJOR << "." << PROJECT_VERSION_MINOR << "." << PROJECT_VERSION_PATCH << std::endl;

    // Declare Variables here
    std::string filePath;
	size_t realSize, frameCount;
    short int bigVoiceBuffer[TRAINSIZE];
    short int littleVoiceBuffer[2000];
    std::vector<std::vector<double> > feature_vector;
    std::vector<std::string> state_label	= { "0", "0", "0", "1", "1", "1", "2", "2", "2", "3", "3", "3", "4", "4", "4", "5", "5", "5", "6", "6", "6", "7", "7", "7", "8", "8", "8", "9", "9", "9", "", "", "" };
	std::string type_covariance		= "diagonal";
	std::string type_model			= "linear";
    int number_states = 9; //state_label.size();
    int number_gaussians = 1, number_iterations = 2;

    // Initialize MFCC
    MFCC mfcc(16000, 25, 10, MFCC::Hamming, 40, 12);
    // Inititalize Clock
    Milliseconds ms;
    // Initialize Datahandler
    DataHandler datahandler;

    // Initialize CDHMM
    CDHMM cdhmm = CDHMM(number_states, type_model, type_covariance, 12, number_gaussians);

    for(int i = 0; i < NUM_WORDS; i++)
    {
        for(int num = 1; num <= 3; num++)
        {
            //** Load wav file
            std::string path = "/Users/timkrebs/OneDrive/Developer/C++/CDHMM/";
            filePath = datahandler.GetFilePath(i, num, 0, "wav");
            filePath = path.append(filePath);

            realSize = datahandler.ReadWav(filePath, bigVoiceBuffer, TRAINSIZE, 0);
            // Check if read operation was successfull
            if(realSize < 1) continue;


            //** Mfcc analyse WITH BIG BUFFER
            frameCount = mfcc.Analyse(bigVoiceBuffer,realSize);
            feature_vector = mfcc.GetMFCCData();

            filePath.erase();
            path = "/Users/timkrebs/OneDrive/Developer/C++/CDHMM/";


            // Initialize and Train
            cdhmm.Initialize(feature_vector[0].size(), number_gaussians, feature_vector.size(), feature_vector);

            for(int j = 0; j < number_iterations; j++)
            {
                // Baum-Welch Algorithmus
                double log_likelihood = cdhmm.Baum_Welch_Algorithm();

            }
        }
    }
    return 0;
}
