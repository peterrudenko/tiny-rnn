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

using namespace TinyRNN;

//SCENARIO("Network can connect with other network", "[network]")
//{
//    GIVEN("Two simple networks")
//    {
//        const int layerSize = 10;
//        
//        const auto network1Name = Uuid::generate();
//        const auto network1 = Network::Prefabs::feedForward(network1Name, layerSize, {layerSize}, layerSize);
//        
//        const auto network2Name = Uuid::generate();
//        const auto network2 = Network::Prefabs::feedForward(network2Name, layerSize, {layerSize}, layerSize);
//        
//        REQUIRE(network1->getName() == network1Name);
//        REQUIRE(network2->getName() == network2Name);
//        REQUIRE(network1->getUuid() != network2->getUuid());
//        
//        WHEN("todo")
//        {
//            // todo
//            
//            THEN("todo")
//            {
//                // todo
//            }
//        }
//    }
//}

//SCENARIO("Network can gate a connection between two other networks", "[network]")
//{
//    GIVEN("Two connected networks and an independent one")
//    {
//        const int layerSize = 10;
//        const auto network1 = Network::Prefabs::feedForward(Uuid::generate(), layerSize, {layerSize}, layerSize);
//        const auto network2 = Network::Prefabs::feedForward(Uuid::generate(), layerSize, {layerSize}, layerSize);
//        const auto network3 = Network::Prefabs::feedForward(Uuid::generate(), layerSize, {layerSize}, layerSize);
//        
//        const auto connections = network1->connectAllToAll(network2);
//        REQUIRE(connections.size() == layerSize * layerSize);
//        
//        WHEN("todo")
//        {
//            // todo
//            
//            THEN("todo")
//            {
//                // todo
//            }
//        }
//    }
//}
