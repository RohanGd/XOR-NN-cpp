// PH05 ROHAN GADGIL
// main.cpp
#include <iostream>
#include "neural_network.hpp"
// #include "matrix.hpp"
#include <vector>
#include <cstdio>


int main()
{
    std::cout << "IMPLEMENTING XOR GATE LOGIC USING ANN." << std::endl;
    // creating neural network
    // 2 input neurons, 3 hidden neurons and 1 output neuron
    std::vector<uint32_t> topology = {2, 13, 1};
    SimpleNeuralNetwork nn(topology, 0.2);

    // sample dataset
    std::vector<std::vector<float>> targetInputs = {
        {0.0f, 0.0f},
        {0.0f, 1.0f},
        {1.0f, 0.0f},
        {1.0f, 1.0f}};
    std::vector<std::vector<float>> targetOutputs = {
        {0.0f},
        {1.0f},
        {1.0f},
        {0.0f}};

    uint32_t trainingSize = 100000;

    // training the neural network with randomized data
    std::cout << "Training started.\nError is calculated as the absolute difference between the target output and the predicted output.\n";

    for (uint32_t i = 0; i < trainingSize; i++)
    {
        uint32_t index = rand() % 4;
        nn.feedForword(targetInputs[index]);
        nn.backPropagate(targetOutputs[index]);

        if (!(i % 5000))
        {
            float error = 0.0f;
            for (int i = 0; i < 4; i++)
            {
                nn.feedForword(targetInputs[i]);
                std::vector<float> preds = nn.getPredictions();
                // std::cout << targetInputs[i][0] << "," << targetInputs[i][1] <<" => " << preds[0] << std::endl;
                error += abs(targetOutputs[i][0] - preds[0]);
            }
            std::cout << "Error percentage: " << std::setprecision(3) << error * 100 << '%' << std::endl;
        }
    }

    std::cout << "Training completed.\nNow testing the neural network.\nPrinting the predictions for the given inputs:\n\n";

//     {
//       std::ofstream outfile("OR_gate_model");
//       boost::archive::text_oarchive archive(outfile);
//       archive << nn;
//    }
    

    // testing the neural network
    for (std::vector<float> input : targetInputs)
    {
        nn.feedForword(input);
        std::vector<float> preds = nn.getPredictions();
        std::cout << input[0] << "," << input[1] << " => " << preds[0] << std::endl;
    }

    return 0;
}
