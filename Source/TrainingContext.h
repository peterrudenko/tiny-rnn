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

#ifndef TINYRNN_LEARNINGCONTEXT_H_INCLUDED
#define TINYRNN_LEARNINGCONTEXT_H_INCLUDED

#include "SerializedObject.h"
#include "SerializationKeys.h"
#include "HardcodedTrainingContext.h"

namespace TinyRNN
{
    class TrainingContext final : public SerializedObject
    {
    public:
        
        using Ptr = std::shared_ptr<TrainingContext>;
        using WeakPtr = std::weak_ptr<TrainingContext>;
        
    public:
        
        class NeuronData final : public SerializedObject
        {
        public:
            
            using Ptr = std::shared_ptr<NeuronData>;
            using Map = std::unordered_map<std::string, NeuronData::Ptr>;
            using SortedMap = std::map<std::string, NeuronData::Ptr>;
            
        public:
            
            explicit NeuronData(const std::string &targetNeuronUuid);
            std::string getNeuronUuid() const noexcept;
            
        public:
            
            virtual void deserialize(SerializationContext::Ptr context) override;
            virtual void serialize(SerializationContext::Ptr context) const override;
            
        private:
            
            double bias;
            double activation;
            double derivative;
            
            double state;
            double oldState;
            
            double errorResponsibility;
            double projectedActivity;
            double gatingActivity;
            
            std::string neuronUuid;
            
            void feedWithRandomBias(double signal);
            void setRandomBias();
            
            friend class Neuron;
            friend class Layer;
            friend class HardcodedNeuron;
            friend class Connection;
            
            TINYRNN_DISALLOW_COPY_AND_ASSIGN(NeuronData);
        };
        
    public:
        
        class ConnectionData final : public SerializedObject
        {
        public:
            
            using Ptr = std::shared_ptr<ConnectionData>;
            using Map = std::unordered_map<std::string, ConnectionData::Ptr>;
            using SortedMap = std::map<std::string, ConnectionData::Ptr>;
            
        public:
            
            explicit ConnectionData(const std::string &targetConnectionUuid);
            std::string getConnectionUuid() const noexcept;
            
        public:
            
            virtual void deserialize(SerializationContext::Ptr context) override;
            virtual void serialize(SerializationContext::Ptr context) const override;
            
        private:
            
            double weight;
            double gain;
            
            std::string connectionUuid;
            
            friend class Neuron;
            friend class HardcodedNeuron;
            friend class Connection;
            
            void setRandomWeight();
            
            TINYRNN_DISALLOW_COPY_AND_ASSIGN(ConnectionData);
        };
        
    public:
        
        explicit TrainingContext(const std::string &name);
        
        std::string getName() const noexcept;
        NeuronData::Ptr getNeuronContext(const std::string &uuid);
        ConnectionData::Ptr getConnectionContext(const std::string &uuid);
        
        void clear();
        
    public:
        
        virtual void deserialize(SerializationContext::Ptr context) override;
        virtual void serialize(SerializationContext::Ptr context) const override;
        
    private:
        
        ConnectionData::Map connectionContexts;
        NeuronData::Map neuronContexts;
        
        std::string name;
        std::string uuid;
        
    private:
        
        TINYRNN_DISALLOW_COPY_AND_ASSIGN(TrainingContext);
    };
    
    
    // =============================================================================
    // TrainingContext implementation
    //
    
    inline TrainingContext::TrainingContext(const std::string &contextName) : name(contextName)
    {
    }
    
    inline std::string TrainingContext::getName() const noexcept
    {
        return this->name;
    }
    
    inline TrainingContext::NeuronData::Ptr TrainingContext::getNeuronContext(const std::string &uuid)
    {
        NeuronData::Ptr neuronContext = this->neuronContexts[uuid];
        
        if (neuronContext != nullptr)
        {
            return neuronContext;
        }
        
        neuronContext = NeuronData::Ptr(new NeuronData(uuid));
        this->neuronContexts[uuid] = neuronContext;
        return neuronContext;
    }
    
    inline TrainingContext::ConnectionData::Ptr TrainingContext::getConnectionContext(const std::string &uuid)
    {
        ConnectionData::Ptr connectionContext = this->connectionContexts[uuid];
        
        if (connectionContext != nullptr)
        {
            return connectionContext;
        }
        
        connectionContext = ConnectionData::Ptr(new ConnectionData(uuid));
        this->connectionContexts[uuid] = connectionContext;
        return connectionContext;
    }

    inline void TrainingContext::clear()
    {
        this->neuronContexts.clear();
        this->connectionContexts.clear();
    }
    
    // =============================================================================
    // Serialization
    //
    
    inline void TrainingContext::deserialize(SerializationContext::Ptr context)
    {
        SerializationContext::Ptr stateNode(context->getChildContext(Serialization::Core::TrainingContext));
        SerializationContext::Ptr root((stateNode != nullptr) ? stateNode : context);
        
        this->uuid = root->getStringProperty(Serialization::Core::Uuid);
        this->name = root->getStringProperty(Serialization::Core::Name);
        
        this->neuronContexts.clear();
        SerializationContext::Ptr neuronStatesNode(root->getChildContext(Serialization::Core::NeuronContexts));
        
        for (size_t i = 0; i < neuronStatesNode->getNumChildrenContexts(); ++i)
        {
            SerializationContext::Ptr neuronStateNode(neuronStatesNode->getChildContext(i));
            NeuronData::Ptr neuronContext(new NeuronData(""));
            neuronContext->deserialize(neuronStateNode);
            this->neuronContexts[neuronContext->getNeuronUuid()] = neuronContext;
        }
        
        this->connectionContexts.clear();
        SerializationContext::Ptr connectionStatesNode(root->getChildContext(Serialization::Core::ConnectionContexts));
        
        for (size_t i = 0; i < connectionStatesNode->getNumChildrenContexts(); ++i)
        {
            SerializationContext::Ptr connectionStateNode(connectionStatesNode->getChildContext(i));
            ConnectionData::Ptr connectionContext(new ConnectionData(""));
            connectionContext->deserialize(connectionStateNode);
            this->connectionContexts[connectionContext->getConnectionUuid()] = connectionContext;
        }
    }
    
    inline void TrainingContext::serialize(SerializationContext::Ptr context) const
    {
        SerializationContext::Ptr stateNode(context->createChildContext(Serialization::Core::TrainingContext));
        
        stateNode->setStringProperty(this->uuid, Serialization::Core::Uuid);
        stateNode->setStringProperty(this->name, Serialization::Core::Name);
        
        SerializationContext::Ptr neuronStatesNode(stateNode->createChildContext(Serialization::Core::NeuronContexts));
        const NeuronData::SortedMap sortedNeuronContexts(this->neuronContexts.begin(), this->neuronContexts.end());
        for (const auto &i : sortedNeuronContexts)
        {
            SerializationContext::Ptr neuronNode(neuronStatesNode->createChildContext(Serialization::Core::TrainingNeuronContext));
            i.second->serialize(neuronNode);
        }
        
        SerializationContext::Ptr connectionStatesNode(stateNode->createChildContext(Serialization::Core::ConnectionContexts));
        const ConnectionData::SortedMap sortedConnectionContexts(this->connectionContexts.begin(), this->connectionContexts.end());
        for (const auto &i : sortedConnectionContexts)
        {
            SerializationContext::Ptr connectionNode(connectionStatesNode->createChildContext(Serialization::Core::TrainingConnectionContext));
            i.second->serialize(connectionNode);
        }
    }
    
    // =============================================================================
    // NeuronData implementation
    //
    
    inline TrainingContext::NeuronData::NeuronData(const std::string &targetNeuronUuid) :
    activation(0.0),
    derivative(0.0),
    state(0.0),
    oldState(0.0),
    errorResponsibility(0.0),
    projectedActivity(0.0),
    gatingActivity(0.0),
    neuronUuid(targetNeuronUuid)
    {
        this->setRandomBias();
    }
    
    inline std::string TrainingContext::NeuronData::getNeuronUuid() const noexcept
    {
        return this->neuronUuid;
    }
    
    inline void TrainingContext::NeuronData::feedWithRandomBias(double signal)
    {
        this->activation = signal;
        this->derivative = 0.0;
        this->setRandomBias();
    }
    
    inline void TrainingContext::NeuronData::setRandomBias()
    {
        std::random_device randomDevice;
        std::mt19937 mt19937(randomDevice());
        std::uniform_real_distribution<double> distribution(-0.1, 0.1);
        this->bias = distribution(mt19937);
    }
    
    inline void TrainingContext::NeuronData::deserialize(SerializationContext::Ptr context)
    {
        this->neuronUuid = context->getStringProperty(Serialization::Core::NeuronUuid);
        this->bias = context->getRealProperty(Serialization::Core::Bias);
        this->activation = context->getRealProperty(Serialization::Core::Activation);
        this->derivative = context->getRealProperty(Serialization::Core::Derivative);
        this->state = context->getRealProperty(Serialization::Core::State);
        this->oldState = context->getRealProperty(Serialization::Core::OldState);
        this->errorResponsibility = context->getRealProperty(Serialization::Core::ErrorResponsibility);
        this->projectedActivity = context->getRealProperty(Serialization::Core::ProjectedActivity);
        this->gatingActivity = context->getRealProperty(Serialization::Core::GatingActivity);
    }
    
    inline void TrainingContext::NeuronData::serialize(SerializationContext::Ptr context) const
    {
        context->setStringProperty(this->neuronUuid, Serialization::Core::NeuronUuid);
        context->setRealProperty(this->bias, Serialization::Core::Bias);
        context->setRealProperty(this->activation, Serialization::Core::Activation);
        context->setRealProperty(this->derivative, Serialization::Core::Derivative);
        context->setRealProperty(this->state, Serialization::Core::State);
        context->setRealProperty(this->oldState, Serialization::Core::OldState);
        context->setRealProperty(this->errorResponsibility, Serialization::Core::ErrorResponsibility);
        context->setRealProperty(this->projectedActivity, Serialization::Core::ProjectedActivity);
        context->setRealProperty(this->gatingActivity, Serialization::Core::GatingActivity);
    }
    
    // =============================================================================
    // ConnectionData implementation
    //
    
    inline TrainingContext::ConnectionData::ConnectionData(const std::string &targetConnectionUuid) :
    weight(0.0),
    gain(1.0),
    connectionUuid(targetConnectionUuid)
    {
        this->setRandomWeight();
    }
    
    inline std::string TrainingContext::ConnectionData::getConnectionUuid() const noexcept
    {
        return this->connectionUuid;
    }
    
    inline void TrainingContext::ConnectionData::setRandomWeight()
    {
        std::random_device randomDevice;
        std::mt19937 mt19937(randomDevice());
        std::uniform_real_distribution<double> distribution(-0.1, 0.1);
        this->weight = distribution(mt19937);
    }
    
    inline void TrainingContext::ConnectionData::deserialize(SerializationContext::Ptr context)
    {
        this->connectionUuid = context->getStringProperty(Serialization::Core::ConnectionUuid);
        this->weight = context->getRealProperty(Serialization::Core::Weight);
        this->gain = context->getRealProperty(Serialization::Core::Gain);
    }
    
    inline void TrainingContext::ConnectionData::serialize(SerializationContext::Ptr context) const
    {
        context->setStringProperty(this->connectionUuid, Serialization::Core::ConnectionUuid);
        context->setRealProperty(this->weight, Serialization::Core::Weight);
        context->setRealProperty(this->gain, Serialization::Core::Gain);
    }
}

#endif // TINYRNN_LEARNINGCONTEXT_H_INCLUDED
