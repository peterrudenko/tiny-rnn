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

#include "ThirdParty/Catch/include/catch.hpp"
#include "ThirdParty/pugixml/src/pugixml.hpp"
#include "Helpers.h"
#include "Serializer.h"
#include "Network.h"
#include "Uuid.h"
#include "ScopedTimer.h"

using namespace TinyRNN;

class XMLSerializationContext final : public SerializationContext
{
public:
    
    using Ptr = std::shared_ptr<XMLSerializationContext>;
    
public:
    
    explicit XMLSerializationContext(pugi::xml_node rootNode) : node(rootNode) {}
    
    virtual void setRealProperty(double value, const std::string &key) override
    { this->node.append_attribute(key.c_str()).set_value(value); }
    
    virtual double getRealProperty(const std::string &key) const override
    { return this->node.attribute(key.c_str()).as_double(); }
    
    virtual void setNumberProperty(long long value, const std::string &key) override
    { this->node.append_attribute(key.c_str()).set_value(value); }
    
    virtual long long getNumberProperty(const std::string &key) const override
    { return this->node.attribute(key.c_str()).as_llong(); }
    
    virtual void setStringProperty(const std::string &value, const std::string &key) override
    { this->node.append_attribute(key.c_str()).set_value(value.c_str()); }
    
    virtual std::string getStringProperty(const std::string &key) const override
    { return std::string(this->node.attribute(key.c_str()).as_string()); }
    
    virtual size_t getNumChildrenContexts() const override
    { return std::distance(this->node.children().begin(), this->node.children().end()); }
    
    virtual SerializationContext::Ptr getChildContext(int index) const override
    {
        pugi::xml_node child = *std::next(this->node.children().begin(), index);
        XMLSerializationContext::Ptr childContext(new XMLSerializationContext(child));
        return childContext;
    }
    
    virtual SerializationContext::Ptr getChildContext(const std::string &key) const override
    {
        pugi::xml_node child = node.child(key.c_str());
        XMLSerializationContext::Ptr childContext(new XMLSerializationContext(child));
        return childContext;
    }
    
    virtual SerializationContext::Ptr createChildContext(const std::string &key) override
    {
        pugi::xml_node newChild = node.append_child(key.c_str());
        XMLSerializationContext::Ptr childContext(new XMLSerializationContext(newChild));
        return childContext;
    }
    
private:
    
    pugi::xml_node node;
    
};

class XMLSerializer final : public Serializer
{
public:
    
    XMLSerializer() {}
    
    virtual ~XMLSerializer() override {}
    
    virtual std::string serialize(SerializedObject::Ptr target) const override
    {
        const ScopedTimer timer("XMLSerializer::serialize");
        
        pugi::xml_document document;
        XMLSerializationContext::Ptr context(new XMLSerializationContext(document.root()));
        target->serialize(context);

        XMLStringWriter writer;
        document.save(writer);
        return writer.result;
    }
    
    virtual void deserialize(SerializedObject::Ptr target, const std::string &data) override
    {
        const ScopedTimer timer("XMLSerializer::deserialize");
        
        pugi::xml_document document;
        pugi::xml_parse_result result = document.load(data.c_str());
        
        if (result)
        {
            XMLSerializationContext::Ptr context(new XMLSerializationContext(document.root()));
            target->deserialize(context);
        }
    }
    
private:
    
    struct XMLStringWriter: pugi::xml_writer
    {
        std::string result;
        
        virtual void write(const void *data, size_t size) override
        {
            result.append(static_cast<const char*>(data), size);
        }
    };
    
};

SCENARIO("Networks can be serialized and deserialized correctly", "[serialization]")
{
    GIVEN("Serialized topology of a randomly trained simple network")
    {
        const int layerSize = RANDOM(10, 20);
        const auto networkName = RANDOMNAME();
        
        const auto network = Network::Prefabs::longShortTermMemory(networkName, 3, {layerSize}, 3);
        REQUIRE(network->getName() == networkName);
        
        const int numTrainingIterations = RANDOM(100, 200);
        for (int i = 0; i < numTrainingIterations; ++i)
        {
            network->feed({ RANDOM(0.0, 10000.0), RANDOM(0.0, 10000.0), RANDOM(0.0, 10000.0) });
            network->train(0.5, { RANDOM(0.0, 10000.0), RANDOM(0.0, 10000.0), RANDOM(0.0, 10000.0) });
        }
        
        XMLSerializer serializer;
        const std::string &serializedTopology = serializer.serialize(network);
        
        WHEN("a new network with the same context is deserialized from that data and serialized back")
        {
            Network::Ptr recreatedNetwork(new Network(network->getContext()));
            serializer.deserialize(recreatedNetwork, serializedTopology);
            const std::string &reserializedTopology = serializer.serialize(recreatedNetwork);
            
            THEN("the serialization results should be equal")
            {
                REQUIRE(serializedTopology == reserializedTopology);
            }
            
            THEN("both networks should produce the same output")
            {
                const int numChecks = RANDOM(10, 20);
                
                for (int i = 0; i < numTrainingIterations; ++i)
                {
                    const double r1 = RANDOM(0.0, 10000.0);
                    const double r2 = RANDOM(0.0, 10000.0);
                    const double r3 = RANDOM(0.0, 10000.0);
                    Neuron::Values result1 = network->feed({r1, r2, r3});
                    Neuron::Values result2 = recreatedNetwork->feed({r1, r2, r3});
                    REQUIRE(result1 == result2);
                }
            }
        }
    }
    
    GIVEN("Serialized training state of a randomly trained simple network")
    {
        const int layerSize = RANDOM(5, 15);
        const auto networkName = RANDOMNAME();
        
        const auto network = Network::Prefabs::feedForward(networkName, 3, {layerSize}, 3);
        REQUIRE(network->getName() == networkName);
        
        const int numTrainingIterations = RANDOM(100, 200);
        for (int i = 0; i < numTrainingIterations; ++i)
        {
            network->feed({ RANDOM(0.0, 10000.0), RANDOM(0.0, 10000.0), RANDOM(0.0, 10000.0) });
            network->train(0.5, { RANDOM(0.0, 10000.0), RANDOM(0.0, 10000.0), RANDOM(0.0, 10000.0) });
        }
        
        XMLSerializer serializer;
        const std::string &serializedTrainingState = serializer.serialize(network->getContext());
        
        WHEN("a new state is deserialized from that data and serialized back")
        {
            TrainingContext::Ptr recreatedContext(new TrainingContext(""));
            serializer.deserialize(recreatedContext, serializedTrainingState);
            const std::string &reserializedState = serializer.serialize(recreatedContext);
            
            THEN("the serialization results should be equal")
            {
                REQUIRE(serializedTrainingState == reserializedState);
            }
        }
    }
}

#if TINYRNN_OPENCL_ACCELERATION

SCENARIO("Hardcoded network can be serialized and deserialized correctly", "[serialization]")
{
    GIVEN("Serialized kernel chunks of a randomly trained hardcoded network")
    {
        const int layerSize = RANDOM(5, 10);
        const auto networkName = RANDOMNAME();
        
        const auto network = Network::Prefabs::longShortTermMemory(networkName, 3, {layerSize}, 3);
        REQUIRE(network->getName() == networkName);
        
        HardcodedNetwork::Ptr clNetwork = network->hardcode();
        clNetwork->compile();
        
        const int numTrainingIterations = RANDOM(100, 200);
        for (int i = 0; i < numTrainingIterations; ++i)
        {
            clNetwork->feed({ RANDOM(0.0, 10000.0), RANDOM(0.0, 10000.0), RANDOM(0.0, 10000.0) });
            clNetwork->train(0.5, { RANDOM(0.0, 10000.0), RANDOM(0.0, 10000.0), RANDOM(0.0, 10000.0) });
        }
        
        XMLSerializer serializer;
        const std::string &serializedTopology = serializer.serialize(clNetwork);
        //std::cout << serializedTopology;
        
        WHEN("a new network with the same context is deserialized from that data and serialized back")
        {
            HardcodedNetwork::Ptr clRecreatedNetwork(new HardcodedNetwork(clNetwork->getContext()));
            serializer.deserialize(clRecreatedNetwork, serializedTopology);
            clRecreatedNetwork->compile();
            const std::string &reserializedTopology = serializer.serialize(clRecreatedNetwork);
            
            THEN("the serialization results should be equal")
            {
                REQUIRE(serializedTopology == reserializedTopology);
            }
            
            THEN("both networks should produce the same output")
            {
                const int numChecks = RANDOM(5, 10);
                
                for (int i = 0; i < numTrainingIterations; ++i)
                {
                    const double r1 = RANDOM(0.0, 10000.0);
                    const double r2 = RANDOM(0.0, 10000.0);
                    const double r3 = RANDOM(0.0, 10000.0);
                    Neuron::Values result1 = clNetwork->feed({r1, r2, r3});
                    Neuron::Values result2 = clRecreatedNetwork->feed({r1, r2, r3});
                    REQUIRE(result1 == result2);
                }
            }
        }
    }
    
    GIVEN("Serialized memory state of a randomly trained hardcoded network")
    {
        const int layerSize = RANDOM(5, 15);
        const auto networkName = RANDOMNAME();
        
        const auto network = Network::Prefabs::longShortTermMemory(networkName, 3, {layerSize}, 3);
        REQUIRE(network->getName() == networkName);
        
        HardcodedNetwork::Ptr clNetwork = network->hardcode();
        clNetwork->compile();
        
        const int numTrainingIterations = RANDOM(100, 200);
        for (int i = 0; i < numTrainingIterations; ++i)
        {
            clNetwork->feed({ RANDOM(0.0, 10000.0), RANDOM(0.0, 10000.0), RANDOM(0.0, 10000.0) });
            clNetwork->train(0.5, { RANDOM(0.0, 10000.0), RANDOM(0.0, 10000.0), RANDOM(0.0, 10000.0) });
        }
        
        XMLSerializer serializer;
        const std::string &serializedMemory = serializer.serialize(clNetwork->getContext());
        //std::cout << serializedMemory;
        
        WHEN("a new state is deserialized from that data and serialized back")
        {
            HardcodedTrainingContext::Ptr recreatedContext(new HardcodedTrainingContext());
            serializer.deserialize(recreatedContext, serializedMemory);
            const std::string &reserializedMemory = serializer.serialize(recreatedContext);
            
            THEN("the serialization results should be equal")
            {
                REQUIRE(serializedMemory == reserializedMemory);
            }
        }
    }
}

#endif
