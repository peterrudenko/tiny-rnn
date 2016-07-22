/*
    Copyright (c) 2016 Peter Rudenko

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

#include "ThirdParty/Catch/include/catch.hpp"
#include "Helpers.h"
#include "Network.h"

using namespace TinyRNN;

SCENARIO("Neurons can be connected with each other", "[neuron]")
{
    GIVEN("Two neurons")
    {
        TrainingContext::Ptr context(new TrainingContext("test"));
        Neuron::Ptr neuron1(new Neuron(context));
        Neuron::Ptr neuron2(new Neuron(context));
        
        REQUIRE(neuron1->getUuid() != neuron2->getUuid());
        
        WHEN("the first neuron is connected to the second one twice")
        {
            Neuron::Connection::Ptr connection1 = neuron1->connectWith(neuron2);
            Neuron::Connection::Ptr connection2 = neuron1->connectWith(neuron2);
            
            THEN("we get the correct connection objects")
            {
                REQUIRE(connection1 != nullptr);
                REQUIRE(connection1 == connection2);
                REQUIRE(!neuron1->isSelfConnected());
                REQUIRE(!neuron2->isSelfConnected());
            }
        }
    }
}

SCENARIO("Layers can be connected all-to-all", "[layer]")
{
    GIVEN("Two layers with some neurons")
    {
        const int numNeurons1 = RANDOM(10, 100);
        const int numNeurons2 = RANDOM(10, 100);
        
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
        const int numNeurons1 = RANDOM(10, 100);
        const int numNeurons2 = numNeurons1 * 2;
        
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
        const int numNeurons = RANDOM(10, 100);
        
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

SCENARIO("Layer can gate a connection between two other layers", "[layer]")
{
    GIVEN("Three equally-sized layers, two of them connected")
    {
        const int numNeurons = RANDOM(10, 100);
        
        TrainingContext::Ptr context(new TrainingContext("test"));
        Layer::Ptr layer1(new Layer(context, numNeurons));
        Layer::Ptr layer2(new Layer(context, numNeurons));
        Layer::Ptr layer3(new Layer(context, numNeurons));
        
        Neuron::Connection::HashMap connections = layer1->connectOneToOne(layer2);
        for (const auto &i : connections)
        {
            REQUIRE(i.second->getGateNeuron() == nullptr);
        }
        
        WHEN("We gate those connections one to one")
        {
            const bool gatedOk = layer3->gateOneToOne(layer1, layer2, connections);
            
            THEN("We make sure connections' gaters are set")
            {
                REQUIRE(gatedOk);
                
                for (const auto &i : connections)
                {
                    REQUIRE(i.second->getGateNeuron() != nullptr);
                }
            }
        }
        
        WHEN("We gate all incoming connections of layer2")
        {
            const bool gatedOk = layer3->gateAllIncomingConnections(layer2, connections);
            
            THEN("We make sure connections' gaters are set")
            {
                REQUIRE(gatedOk);
                
                for (const auto &i : connections)
                {
                    REQUIRE(i.second->getGateNeuron() != nullptr);
                }
            }
        }
        
        WHEN("We gate all outgoing connections of layer1")
        {
            const bool gatedOk = layer3->gateAllOutgoingConnections(layer1, connections);
            
            THEN("We make sure connections' gaters are set")
            {
                REQUIRE(gatedOk);
                
                for (const auto &i : connections)
                {
                    REQUIRE(i.second->getGateNeuron() != nullptr);
                }
            }
        }
    }
    
    GIVEN("Three different-sized layers")
    {
        const int numNeurons = RANDOM(10, 100);
        
        TrainingContext::Ptr context(new TrainingContext("test"));
        Layer::Ptr layer1(new Layer(context, numNeurons));
        Layer::Ptr layer2(new Layer(context, numNeurons + 10));
        Layer::Ptr layer3(new Layer(context, numNeurons + 20));
        Neuron::Connection::HashMap connections = layer1->connectAllToAll(layer2);
        
        WHEN("We try to gate those connections")
        {
            const bool gate1Ok = layer3->gateOneToOne(layer1, layer2, connections);
            const bool gate2Ok = layer3->gateAllIncomingConnections(layer2, connections);
            const bool gate3Ok = layer3->gateAllOutgoingConnections(layer1, connections);
            
            THEN("We make sure the gating failed")
            {
                REQUIRE(!gate1Ok);
                REQUIRE(!gate2Ok);
                REQUIRE(!gate3Ok);
            }
        }
    }
}
