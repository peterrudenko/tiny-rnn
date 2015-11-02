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
            
            THEN("we get the same connection object")
            {
                REQUIRE(connection1 == connection2);
            }
        }
    }
}

//SCENARIO("Neurons can have a self-connection", "[neuron]")
//{
//    // todo
//}
