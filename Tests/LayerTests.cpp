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
#include "TinyRNN.h"

using namespace TinyRNN;

SCENARIO("Layers can be connected all-to-all", "[layer]")
{
    GIVEN("Two layers with some neurons")
    {
        std::random_device randomDevice;
        std::mt19937 mt19937(randomDevice());
        std::uniform_int_distribution<unsigned int> distribution(10, 100);
        const unsigned int numNeurons1 = distribution(mt19937);
        const unsigned int numNeurons2 = distribution(mt19937);
        
        TrainingContext::Ptr context(new TrainingContext("test"));
        Layer::Ptr layer1(new Layer(context, numNeurons1));
        Layer::Ptr layer2(new Layer(context, numNeurons2));
        
        REQUIRE(layer1->getSize() == numNeurons1);
        REQUIRE(layer1->getSelfConnections().empty());
        
        REQUIRE(layer2->getSize() == numNeurons2);
        REQUIRE(layer2->getSelfConnections().empty());
        
        WHEN("The first layer is all-to-all connected with the second one")
        {
            const auto &connections = layer1->connectAllToAll(layer2);
            INFO(connections.size());
            
            THEN("The number of connections is correct")
            {
                REQUIRE(connections.size() == layer1->getSize() * layer2->getSize());
            }
        }
    }
}

SCENARIO("Layers can be connected one to one", "[layer]")
{
    GIVEN("Two layers with different sizes")
    {
        std::random_device randomDevice;
        std::mt19937 mt19937(randomDevice());
        std::uniform_int_distribution<unsigned int> distribution(10, 100);
        const unsigned int numNeurons1 = distribution(mt19937);
        const unsigned int numNeurons2 = numNeurons1 * 2;
        
        TrainingContext::Ptr context(new TrainingContext("test"));
        Layer::Ptr layer1(new Layer(context, numNeurons1));
        Layer::Ptr layer2(new Layer(context, numNeurons2));
        
        REQUIRE(layer1->getSize() == numNeurons1);
        REQUIRE(layer1->getSelfConnections().empty());
        
        REQUIRE(layer2->getSize() == numNeurons2);
        REQUIRE(layer2->getSelfConnections().empty());

        WHEN("We try to connect them one-to-one")
        {
            const auto &connections = layer1->connectOneToOne(layer2);
            
            THEN("There are no connections made")
            {
                REQUIRE(connections.empty());
            }
        }
    }
    
    GIVEN("Two layers with the same sizes")
    {
        std::random_device randomDevice;
        std::mt19937 mt19937(randomDevice());
        std::uniform_int_distribution<unsigned int> distribution(10, 100);
        const unsigned int numNeurons = distribution(mt19937);
        
        TrainingContext::Ptr context(new TrainingContext("test"));
        Layer::Ptr layer1(new Layer(context, numNeurons));
        Layer::Ptr layer2(new Layer(context, numNeurons));
        
        REQUIRE(layer1->getSize() == numNeurons);
        REQUIRE(layer1->getSelfConnections().empty());
        
        REQUIRE(layer2->getSize() == numNeurons);
        REQUIRE(layer2->getSelfConnections().empty());
        
        WHEN("We try to connect them one-to-one")
        {
            const auto &connections = layer1->connectOneToOne(layer2);
            
            THEN("The number of connections is correct")
            {
                REQUIRE(connections.size() == numNeurons);
            }
        }
    }
}

//SCENARIO("Layer can gate a connection between two other layers", "[layer]")
//{
//    // todo
//}
