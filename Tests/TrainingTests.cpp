/*
    Copyright (c) 2015 Peter Rudenko

    Permission is hereby granted, free of charge, to any person obtaining
    a copy of this software and associated documentation files (the "Software"),
    to deal in the Software without restriction, including without limitation
    the rights to use, copy, modify, merge, publish, distribute, sublicense,
    and/or sell copies of the Software, and to permit persons to whom the Software
    is furnished to do so, subject to the following conditions:

    The above copyright notice and this permission notice shall be included in
    all copies or substantial portions of the Software.

    THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
    EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES
    OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.
    IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY
    CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT,
    TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE
    OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
*/

#include "ThirdParty/catch.hpp"
#include "Network.h"
#include <float.h>

using namespace TinyRNN;

SCENARIO("A perceptron can be trained to model a xor function", "[training]")
{
    GIVEN("A single-layer perceptron")
    {
        std::random_device randomDevice;
        std::mt19937 mt19937(randomDevice());
        std::uniform_int_distribution<unsigned int> distribution(1500, 2000);
        
        const int numIterations = distribution(mt19937);
        
        const auto contextName = Uuid::generate();
        const auto networkName = Uuid::generate();
        
        TrainingContext::Ptr context(new TrainingContext(contextName));
        REQUIRE(context->getName() == contextName);
        
        Layer::Ptr inputLayer(new Layer(context, 2));
        Layer::Ptr hiddenLayer(new Layer(context, 10));
        Layer::Ptr outputLayer(new Layer(context, 1));
        
        REQUIRE(inputLayer->getUuid() != hiddenLayer->getUuid());
        REQUIRE(hiddenLayer->getUuid() != outputLayer->getUuid());
        REQUIRE(outputLayer->getUuid() != inputLayer->getUuid());
        
        inputLayer->connectAllToAll(hiddenLayer);
        hiddenLayer->connectAllToAll(outputLayer);
        
        Network::Ptr network(new Network(networkName, context, inputLayer, {hiddenLayer}, outputLayer));
        REQUIRE(network->getName() == networkName);
        REQUIRE(network->getContext() == context);
        
        WHEN("the network is trained with some random number of iterations (from 1500 to 2000)")
        {
            {
                const ScopedTimer timer("Training usual network");
                double rate = 0.5;
                
                for (int i = 0; i < numIterations; ++i)
                {
                    const auto result1 = network->feed({0.0, 1.0});
                    network->train(rate, {1.0});
                    
                    const auto result2 = network->feed({1.0, 0.0});
                    network->train(rate, {1.0});
                    
                    const auto result3 = network->feed({0.0, 0.0});
                    network->train(rate, {0.0});
                    
                    const auto result4 = network->feed({1.0, 1.0});
                    network->train(rate, {0.0});
                }
            }
            
            THEN("it gives a reasonable output")
            {
                const auto result1 = network->feed({0.0, 1.0});
                REQUIRE(result1.size() == 1);
                INFO(result1.front());
                REQUIRE(result1.front() > 0.9);
                
                const auto result2 = network->feed({1.0, 0.0});
                REQUIRE(result2.size() == 1);
                INFO(result2.front());
                REQUIRE(result2.front() > 0.9);
                
                const auto result3 = network->feed({0.0, 0.0});
                REQUIRE(result3.size() == 1);
                INFO(result3.front());
                REQUIRE(result3.front() < 0.1);
                
                const auto result4 = network->feed({1.0, 1.0});
                REQUIRE(result4.size() == 1);
                INFO(result4.front());
                REQUIRE(result4.front() < 0.1);
            }
        }
        
#if TINYRNN_OPENCL_ACCELERATION
        
        WHEN("the hardcoded network is trained with some random number of iterations (from 500 to 1000)")
        {
            network->getContext()->clear();
            
            HardcodedNetwork::Ptr clNetwork = network->hardcode();
            clNetwork->compile();
            
            {
                const ScopedTimer timer("Training hardcoded network");
                double rate = 0.5;
                
                for (int i = 0; i < numIterations; ++i)
                {
                    const auto result1 = clNetwork->feed({0.0, 1.0});
                    clNetwork->train(rate, {1.0});
                    
                    const auto result2 = clNetwork->feed({1.0, 0.0});
                    clNetwork->train(rate, {1.0});
                    
                    const auto result3 = clNetwork->feed({0.0, 0.0});
                    clNetwork->train(rate, {0.0});
                    
                    const auto result4 = clNetwork->feed({1.0, 1.0});
                    clNetwork->train(rate, {0.0});
                }
            }
            
            THEN("it gives a reasonable output")
            {
                const auto result1 = clNetwork->feed({0.0, 1.0});
                REQUIRE(result1.size() == 1);
                INFO(result1.front());
                REQUIRE(result1.front() > 0.9);
                
                const auto result2 = clNetwork->feed({1.0, 0.0});
                REQUIRE(result2.size() == 1);
                INFO(result2.front());
                REQUIRE(result2.front() > 0.9);
                
                const auto result3 = clNetwork->feed({0.0, 0.0});
                REQUIRE(result3.size() == 1);
                INFO(result3.front());
                REQUIRE(result3.front() < 0.1);
                
                const auto result4 = clNetwork->feed({1.0, 1.0});
                REQUIRE(result4.size() == 1);
                INFO(result4.front());
                REQUIRE(result4.front() < 0.1);
            }
        }
        
#endif
        
    }
}

// todo
template<class T>
class CrossEntropyTrainingSession final
{
public:
    
    CrossEntropyTrainingSession(typename T::Ptr targetNetwork, double initialRate = 0.5) :
    network(targetNetwork),
    rate(initialRate)
    {
    }
    
    
    
    double iterate(const Neuron::Values &inputs,
                   const Neuron::Values &targets,
                   const Neuron::Values &rate) const
    {
        const auto &outputs = this->network->feed(inputs);
        this->network->train(rate, targets);
        double error = this->cost(targets, outputs);
        return error;
    }
    
    double cost(const Neuron::Values &targets,
                const Neuron::Values &outputs) const
    {
        double cost = 0.0;
        
        for (size_t i = 0; i < outputs.size(); ++i)
        {
            cost -= (targets[i] * log(outputs[i] + DBL_MIN)) + ((1 - targets[i]) * log(DBL_MIN - outputs[i]));
        }
        
        return cost;
    }
    
private:
    
    typename T::Ptr network;
    
    double rate;
    
};

//SCENARIO("A perceptron can be trained with a custom function", "[training]")
//{
//    GIVEN("A multi-layer perceptron")
//    {
//        std::random_device randomDevice;
//        std::mt19937 mt19937(randomDevice());
//        std::uniform_int_distribution<unsigned int> numIterationsDistribution(10, 100);
//        std::uniform_int_distribution<unsigned int> numHiddenLayersDistribution(1000, 5000);
//
//        const int numIterations = numIterationsDistribution(mt19937);
//        const int numHiddenLayers = numHiddenLayersDistribution(mt19937);
//        
//        const auto contextName = Uuid::generate();
//        const auto networkName = Uuid::generate();
//        
//        TrainingContext::Ptr context(new TrainingContext(contextName));
//        REQUIRE(context->getName() == contextName);
//        
//        std::vector<Layer::Ptr> hiddenLayers;
//        Layer::Ptr prevHiddenLayer = nullptr;
//        Layer::Ptr inputLayer(new Layer(context, 2));
//        
//        for (int i = 0; i < numHiddenLayers; ++i)
//        {
//            Layer::Ptr hiddenLayer(new Layer(context, 10));
//            
//            if (i == 0)
//            {
//                inputLayer->connectAllToAll(hiddenLayer);
//            }
//            else if (prevHiddenLayer != nullptr)
//            {
//                prevHiddenLayer->connectAllToAll(hiddenLayer);
//            }
//            
//            prevHiddenLayer = hiddenLayer;
//            hiddenLayers.push_back(hiddenLayer);
//        }
//        
//        Layer::Ptr outputLayer(new Layer(context, 1));
//        prevHiddenLayer->connectAllToAll(outputLayer);
//        
//        Network::Ptr network(new Network(networkName, context, inputLayer, hiddenLayers, outputLayer));
//        REQUIRE(network->getName() == networkName);
//        REQUIRE(network->getContext() == context);
//        
//        WHEN("the network is trained with some random number of iterations (from 1000 to 5000)")
//        {
//            // todo
//            
//            THEN("it gives a reasonable output")
//            {
//                // todo
//            }
//        }
//    }
//}

