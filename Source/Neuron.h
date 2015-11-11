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

#ifndef TINYRNN_NEURON_H_INCLUDED
#define TINYRNN_NEURON_H_INCLUDED

#include "SerializedObject.h"
#include "TrainingContext.h"
#include "SerializationKeys.h"
#include "Uuid.h"

namespace TinyRNN
{
    class TrainingContext;
    
    class Neuron final : public SerializedObject,
                         public std::enable_shared_from_this<Neuron>
    {
    public:
        
        using Ptr = std::shared_ptr<Neuron>;
        using WeakPtr = std::weak_ptr<Neuron>;
        using Map = std::unordered_map<std::string, Neuron::Ptr>;
        using Array = std::vector<Neuron::Ptr>;
        using Values = std::vector<double>;
        
    public:
        
        class Connection : public SerializedObject,
                           public std::enable_shared_from_this<Connection>
        {
        public:
            
            using Ptr = std::shared_ptr<Connection>;
            using Map = std::unordered_map<std::string, Connection::Ptr>;
            using SortedMap = std::map<std::string, Connection::Ptr>;
            
        public:
            
            explicit Connection(TrainingContext::Ptr context);
            
            Connection(TrainingContext::Ptr context,
                       Neuron::WeakPtr input,
                       Neuron::WeakPtr output);
            
            std::string getUuid() const noexcept;
            TrainingContext::ConnectionData::Ptr getTrainingData() const;
            
            std::string getInputNeuronUuid() const noexcept;
            std::string getOutputNeuronUuid() const noexcept;
            std::string getGateNeuronUuid() const noexcept;
            
            Neuron::Ptr getInputNeuron() const;
            Neuron::Ptr getGateNeuron() const;
            Neuron::Ptr getOutputNeuron() const;
            
            bool hasGate() const noexcept;
            void setGate(Neuron::WeakPtr gateNeuron);
            void setInputOutput(Neuron::WeakPtr inputNeuron, Neuron::WeakPtr outputNeuron);
            
        public:
            
            virtual void deserialize(SerializationContext::Ptr context) override;
            virtual void serialize(SerializationContext::Ptr context) const override;
            
        private:
            
            std::string uuid;
            
            std::string inputNeuronUuid;
            std::string gateNeuronUuid;
            std::string outputNeuronUuid;
            
            Neuron::WeakPtr inputNeuron;
            Neuron::WeakPtr gateNeuron;
            Neuron::WeakPtr outputNeuron;
            
            TrainingContext::Ptr context;
            
        private:
            
            TINYRNN_DISALLOW_COPY_AND_ASSIGN(Connection);
        };
        
    public:
        
        explicit Neuron(TrainingContext::Ptr context);
        Neuron(TrainingContext::Ptr context, double defaultBias);
        
        std::string getUuid() const noexcept;
        TrainingContext::NeuronData::Ptr getTrainingData() const;
        
        Connection::Map getOutgoingConnections() const;
        
        
        bool isSelfConnected() const noexcept;
        bool isConnectedTo(Neuron::Ptr other) const;
        Connection::Ptr getSelfConnection() const noexcept;
        
        Connection::Ptr findConnectionWith(Neuron::Ptr other) const;
        Connection::Ptr findOutgoingConnectionTo(Neuron::Ptr other) const;
        Connection::Ptr findIncomingConnectionFrom(Neuron::Ptr other) const;
        
        Connection::Ptr connectWith(Neuron::Ptr other);
        
        void gate(Connection::Ptr connection);
        
        // Used for the neurons in input layer
        void feed(double value);
        
        // Used for all layers other that input
        double process();
        
        // Used for the neurons in output layer
        void train(double rate, double target);
        
        // Used for all layers other that input
        void backPropagate(double rate);
        
    public:
        
        virtual void deserialize(SerializationContext::Ptr context) override;
        virtual void serialize(SerializationContext::Ptr context) const override;
        
    private:
        
        std::string uuid;
        
        static double activation(double x);
        static double derivative(double x);
        
        Connection::Map incomingConnections;
        Connection::Map outgoingConnections;
        Connection::Map gatedConnections;
        Connection::Ptr selfConnection;
        
        TrainingContext::Ptr context;
        
        bool isOutput() const;
        void learn(double rate = 0.1);
        
        friend class Layer;
        friend class HardcodedNeuron;
        
    private:
        
        // The cache maps, never serialized
        using EligibilityMap = std::unordered_map<std::string, double>;
        using ExtendedEligibilityMap = std::unordered_map<std::string, EligibilityMap>;
        using Influences = std::unordered_map<std::string, Connection::Ptr>;
        using InfluencesMap = std::unordered_map<std::string, Influences>;
        
        mutable InfluencesMap influences;
        mutable EligibilityMap eligibility;
        mutable ExtendedEligibilityMap extended;
        
        mutable Neuron::Map neighbours;
        
    private:
        
        TINYRNN_DISALLOW_COPY_AND_ASSIGN(Neuron);
    };
    
    
    // =============================================================================
    // Neuron implementation
    //
    
    inline Neuron::Neuron(TrainingContext::Ptr targetContext) :
    uuid(Uuid::generate()),
    context(targetContext)
    {
    }
    
    inline Neuron::Neuron(TrainingContext::Ptr targetContext,
                          double defaultBias) :
    uuid(Uuid::generate()),
    context(targetContext)
    {
    }
    
    inline std::string Neuron::getUuid() const noexcept
    {
        return this->uuid;
    }
    
    inline TrainingContext::NeuronData::Ptr Neuron::getTrainingData() const
    {
        return this->context->getNeuronContext(this->getUuid());
    }
    
    // =============================================================================
    // Connections
    //
    
    inline Neuron::Connection::Map Neuron::getOutgoingConnections() const
    {
        Connection::Map outgoing;
        outgoing.insert(this->outgoingConnections.begin(), this->outgoingConnections.end());
        
        if (this->isSelfConnected())
        {
            outgoing[this->selfConnection->getUuid()] = this->selfConnection;
        }
        
        return outgoing;
    }
    
    inline Neuron::Connection::Ptr Neuron::connectWith(Neuron::Ptr other)
    {
        if (other.get() == this)
        {
            this->selfConnection = Connection::Ptr(new Connection(this->context,
                                                                  this->shared_from_this(),
                                                                  this->shared_from_this()));
            return this->selfConnection;
        }
        
        if (Connection::Ptr existingOutgoingConnection = this->findOutgoingConnectionTo(other))
        {
            return existingOutgoingConnection;
        }
        
        Connection::Ptr newConnection(new Connection(this->context, this->shared_from_this(), other));
        const std::string newConnectionId = newConnection->getUuid();
        
        // reference all the connections and traces
        this->outgoingConnections[newConnectionId] = newConnection;
        this->neighbours[other->getUuid()] = other;
        
        other->incomingConnections[newConnectionId] = newConnection;
        other->eligibility[newConnectionId] = 0.0;
        
        for (auto &extendedTrace : other->extended)
        {
            extendedTrace.second[newConnectionId] = 0.0;
        }
        
        return newConnection;
    }
    
    inline void Neuron::gate(Connection::Ptr connection)
    {
        const std::string connectionId = connection->getUuid();
        auto myState = this->getTrainingData();
        
        // add connection to gated list
        this->gatedConnections[connectionId] = connection;
        
        Neuron::Ptr targetNeuron = connection->getOutputNeuron();
        
        const bool targetNeuronNotFoundInExtendedTrace = (this->extended.find(targetNeuron->getUuid()) == this->extended.end());
        if (targetNeuronNotFoundInExtendedTrace)
        {
            // extended trace
            this->neighbours[targetNeuron->getUuid()] = targetNeuron;
            
            EligibilityMap &xtrace = this->extended[targetNeuron->getUuid()];
            xtrace.clear();
            
            for (auto &i : this->incomingConnections)
            {
                Connection::Ptr input = i.second;
                xtrace[input->getUuid()] = 0.0;
            }
        }
        
        this->influences[targetNeuron->getUuid()][connection->getUuid()] = connection;
        
        // set gater
        connection->setGate(this->shared_from_this());
    }
    
    // =============================================================================
    // Core
    //
    
    inline void Neuron::feed(double signalValue)
    {
        const bool noInputConnections = this->incomingConnections.empty();
        const bool hasOutputConnections = !this->outgoingConnections.empty();
        const bool isInputNeuron = (noInputConnections && hasOutputConnections);
        
        if (isInputNeuron)
        {
            this->getTrainingData()->feedWithRandomBias(signalValue);
        }
    }
    
    inline double Neuron::process()
    {
        auto myData = this->getTrainingData();
        myData->oldState = myData->state;
        
        // eq. 15
        if (this->isSelfConnected())
        {
            auto selfConnectionData = this->selfConnection->getTrainingData();
            myData->state = selfConnectionData->gain * selfConnectionData->weight * myData->state + myData->bias;
        }
        else
        {
            myData->state = myData->bias;
        }
        
        for (auto &i : this->incomingConnections)
        {
            const Connection::Ptr inputConnection = i.second;
            const auto inputConnectionData = inputConnection->getTrainingData();
            const auto inputNeuronData = inputConnection->getInputNeuron()->getTrainingData();
            myData->state += inputNeuronData->activation * inputConnectionData->weight * inputConnectionData->gain;
        }
        
        // eq. 16
        myData->activation = Neuron::activation(myData->state);
        
        // f'(s)
        myData->derivative = Neuron::derivative(myData->state);
        
        // update traces
        EligibilityMap influences;
        for (auto &id : this->extended)
        {
            // extended elegibility trace
            Neuron::Ptr neighbour = this->neighbours[id.first];
            
            double influence = 0.0;
            
            // if gated neuron's selfconnection is gated by this unit, the influence keeps track of the neuron's old state
            if (Connection::Ptr neighbourSelfconnection = neighbour->getSelfConnection())
            {
                if (neighbourSelfconnection->getGateNeuron().get() == this)
                {
                    const auto neighbourData = neighbour->getTrainingData();
                    influence = neighbourData->oldState;
                }
            }
            
            // index runs over all the incoming connections to the gated neuron that are gated by this unit
            for (auto &incoming : this->influences[neighbour->getUuid()])
            { // captures the effect that has an input connection to this unit, on a neuron that is gated by this unit
                const Connection::Ptr inputConnection = incoming.second;
                const auto inputConnectionData = inputConnection->getTrainingData();
                const auto inputNeuronData = inputConnection->getInputNeuron()->getTrainingData();
                influence += inputConnectionData->weight * inputNeuronData->activation;
            }
            
            influences[neighbour->getUuid()] = influence;
        }
        
        for (auto &i : this->incomingConnections)
        {
            const Connection::Ptr inputConnection = i.second;
            
            // elegibility trace - Eq. 17
            const double oldElegibility = this->eligibility[inputConnection->getUuid()];
            const auto inputConnectionData = inputConnection->getTrainingData();
            const auto inputNeuronData = inputConnection->getInputNeuron()->getTrainingData();
            this->eligibility[inputConnection->getUuid()] = inputConnectionData->gain * inputNeuronData->activation;
            
            if (this->isSelfConnected())
            {
                const auto selfConnectionData = this->selfConnection->getTrainingData();
                this->eligibility[inputConnection->getUuid()] += selfConnectionData->gain * selfConnectionData->weight * oldElegibility;
            }
            
            for (auto &i : this->extended)
            {
                // extended elegibility trace
                const std::string neuronId = i.first;
                const double influence = influences[neuronId];
                EligibilityMap &xtrace = i.second;
                Neuron::Ptr neighbour = this->neighbours[neuronId];
                
                const auto neighbourData = neighbour->getTrainingData();
                
                // eq. 18
                const double oldXTrace = xtrace[inputConnection->getUuid()];
                xtrace[inputConnection->getUuid()] = myData->derivative * this->eligibility[inputConnection->getUuid()] * influence;
                
                if (Connection::Ptr neighbourSelfConnection = neighbour->getSelfConnection())
                {
                    const auto neighbourSelfConnectionData = neighbourSelfConnection->getTrainingData();
                    xtrace[inputConnection->getUuid()] += neighbourSelfConnectionData->gain * neighbourSelfConnectionData->weight * oldXTrace;
                }
            }
        }
        
        // update gated connection's gains
        for (auto &i : this->gatedConnections)
        {
            const Connection::Ptr connection = i.second;
            auto connectionData = connection->getTrainingData();
            connectionData->gain = myData->activation;
        }
        
        return myData->activation;
    }
    
    inline bool Neuron::isOutput() const
    {
        const bool noProjections = this->outgoingConnections.empty();
        const bool noGates = this->gatedConnections.empty();
        const bool isOutput = (noProjections && noGates);
        return isOutput;
    }
    
    inline void Neuron::train(double rate, double target)
    {
        // output neurons get their error from the enviroment
        if (this->isOutput())
        {
            auto myData = this->getTrainingData();
            myData->errorResponsibility = myData->projectedActivity = target - myData->activation; // Eq. 10
            this->learn(rate);
        }
    }
    
    inline void Neuron::backPropagate(double rate)
    {
        double errorAccumulator = 0.0;
        
        // the rest of the neuron compute their error responsibilities by backpropagation
        if (! this->isOutput())
        {
            auto myData = this->getTrainingData();
            
            // error responsibilities from all the connections projected from this neuron
            for (auto &i : this->outgoingConnections)
            {
                const Connection::Ptr connection = i.second;
                const auto outputConnectionData = connection->getTrainingData();
                const auto outputNeuronData = connection->getOutputNeuron()->getTrainingData();
                // Eq. 21
                errorAccumulator += outputNeuronData->errorResponsibility * outputConnectionData->gain * outputConnectionData->weight;
            }
            
            // projected error responsibility
            myData->projectedActivity = myData->derivative * errorAccumulator;
            
            errorAccumulator = 0.0;
            
            // error responsibilities from all the connections gated by this neuron
            for (auto &i : this->extended)
            {
                const std::string gatedNeuronId = i.first;
                const Neuron::Ptr gatedNeuron = this->neighbours[gatedNeuronId];
                const auto gatedNeuronData = gatedNeuron->getTrainingData();
                
                double influence = 0.0;
                
                // if gated neuron's selfconnection is gated by this neuron
                if (Connection::Ptr gatedNeuronSelfConnection = gatedNeuron->getSelfConnection())
                {
                    if (gatedNeuronSelfConnection->getGateNeuron().get() == this)
                    {
                        influence = gatedNeuronData->oldState;
                    }
                }
                
                // index runs over all the connections to the gated neuron that are gated by this neuron
                for (auto &i : this->influences[gatedNeuronId])
                { // captures the effect that the input connection of this neuron have, on a neuron which its input/s is/are gated by this neuron
                    const Connection::Ptr inputConnection = i.second;
                    const auto inputConnectionData = inputConnection->getTrainingData();
                    const auto inputNeuronData = inputConnection->getInputNeuron()->getTrainingData();
                    influence += inputConnectionData->weight * inputNeuronData->activation;
                }
                
                // eq. 22
                errorAccumulator += gatedNeuronData->errorResponsibility * influence;
            }
            
            // gated error responsibility
            myData->gatingActivity = myData->derivative * errorAccumulator;
            
            // error responsibility - Eq. 23
            myData->errorResponsibility = myData->projectedActivity + myData->gatingActivity;
            
            this->learn(rate);
        }
    }
    
    inline void Neuron::learn(double rate)
    {
        auto myData = this->getTrainingData();
        
        // adjust all the neuron's incoming connections
        for (auto &i : this->incomingConnections)
        {
            const std::string inputConnectionUuid = i.first;
            const Connection::Ptr inputConnection = i.second;
            
            // Eq. 24
            double gradient = myData->projectedActivity * this->eligibility[inputConnectionUuid];
            for (auto &ext : this->extended)
            {
                const std::string neuronUuid = ext.first;
                const auto neuronData = this->context->getNeuronContext(neuronUuid);
                gradient += neuronData->errorResponsibility * this->extended[neuronUuid][inputConnectionUuid];
            }
            
            auto inputConnectionData = inputConnection->getTrainingData();
            inputConnectionData->weight += rate * gradient; // adjust weights - aka learn
        }
        
        // adjust bias
        myData->bias += rate * myData->errorResponsibility;
    }
    
    inline double Neuron::activation(double x)
    {
        return 1.0 / (1.0 + exp(-x));
    }
    
    inline double Neuron::derivative(double x)
    {
        const double fx = Neuron::activation(x);
        return fx * (1.0 - fx);
    }
    
    
    // =============================================================================
    // Const stuff
    //
    
    inline bool Neuron::isSelfConnected() const noexcept
    {
        return (this->selfConnection != nullptr);
    }
    
    inline bool Neuron::isConnectedTo(Neuron::Ptr other) const
    {
        return (this->findConnectionWith(other) != nullptr);
    }
    
    inline Neuron::Connection::Ptr Neuron::getSelfConnection() const noexcept
    {
        return this->selfConnection;
    }
    
    inline Neuron::Connection::Ptr Neuron::findConnectionWith(Neuron::Ptr other) const
    {
        if (other.get() == this)
        {
            return this->selfConnection;
        }
        
        for (const auto &i : this->incomingConnections)
        {
            const Connection::Ptr connection = i.second;
            
            if (connection->getInputNeuron() == other)
            {
                return connection;
            }
        }
        
        for (const auto &i : this->outgoingConnections)
        {
            const Connection::Ptr connection = i.second;
            
            if (connection->getOutputNeuron() == other)
            {
                return connection;
            }
        }
        
        for (const auto &i : this->gatedConnections)
        {
            const Connection::Ptr connection = i.second;
            
            if (connection->getInputNeuron() == other ||
                connection->getOutputNeuron() == other)
            {
                return connection;
            }
        }
        
        return nullptr;
    }
    
    inline Neuron::Connection::Ptr Neuron::findOutgoingConnectionTo(Neuron::Ptr other) const
    {
        for (const auto &i : this->outgoingConnections)
        {
            const Neuron::Connection::Ptr connection = i.second;
            
            if (connection->getOutputNeuron() == other)
            {
                return connection;
            }
        }
        
        return nullptr;
    }
    
    inline Neuron::Connection::Ptr Neuron::findIncomingConnectionFrom(Neuron::Ptr other) const
    {
        for (const auto &i : this->incomingConnections)
        {
            const Neuron::Connection::Ptr connection = i.second;
            
            if (connection->getInputNeuron() == other)
            {
                return connection;
            }
        }
        
        return nullptr;
    }
    
    // =============================================================================
    // Serialization
    //
    
    inline void Neuron::deserialize(SerializationContext::Ptr context)
    {
        this->uuid = context->getStringProperty(Keys::Core::Uuid);
        // selfconnection will be restored in network deserialization
    }
    
    inline void Neuron::serialize(SerializationContext::Ptr context) const
    {
        context->setStringProperty(this->uuid, Keys::Core::Uuid);
    }
    
    // =============================================================================
    // Connection
    //
    
    inline Neuron::Connection::Connection(TrainingContext::Ptr targetContext) :
    uuid(Uuid::generate()),
    context(targetContext)
    {
    }
    
    inline Neuron::Connection::Connection(TrainingContext::Ptr targetContext,
                                   std::weak_ptr<Neuron> input,
                                   std::weak_ptr<Neuron> output) :
    uuid(Uuid::generate()),
    inputNeuronUuid(input.lock()->getUuid()),
    outputNeuronUuid(output.lock()->getUuid()),
    inputNeuron(input),
    outputNeuron(output),
    context(targetContext)
    {
        //
    }
    
    inline TrainingContext::ConnectionData::Ptr Neuron::Connection::getTrainingData() const
    {
        return this->context->getConnectionContext(this->getUuid());
    }
    
    inline std::string Neuron::Connection::getUuid() const noexcept
    {
        return this->uuid;
    }
    
    inline std::string Neuron::Connection::getInputNeuronUuid() const noexcept
    {
        return this->inputNeuronUuid;
    }
    
    inline std::string Neuron::Connection::getOutputNeuronUuid() const noexcept
    {
        return this->outputNeuronUuid;
    }
    
    inline std::string Neuron::Connection::getGateNeuronUuid() const noexcept
    {
        return this->gateNeuronUuid;
    }
    
    inline Neuron::Ptr Neuron::Connection::getInputNeuron() const
    {
        return this->inputNeuron.lock();
    }
    
    inline Neuron::Ptr Neuron::Connection::getGateNeuron() const
    {
        return this->gateNeuron.lock();
    }
    
    inline Neuron::Ptr Neuron::Connection::getOutputNeuron() const
    {
        return this->outputNeuron.lock();
    }
    
    inline bool Neuron::Connection::hasGate() const noexcept
    {
        return ! this->gateNeuronUuid.empty();
    }
    
    inline void Neuron::Connection::setGate(Neuron::WeakPtr gateNeuron)
    {
        this->gateNeuron = gateNeuron;
        this->gateNeuronUuid = gateNeuron.lock()->getUuid();
    }
    
    inline void Neuron::Connection::setInputOutput(Neuron::WeakPtr weakInput, Neuron::WeakPtr weakOutput)
    {
        Neuron::Ptr strongInput = weakInput.lock();
        Neuron::Ptr strongOutput = weakOutput.lock();
        
        this->inputNeuron = strongInput;
        this->outputNeuron = strongOutput;
        this->inputNeuronUuid = strongInput->getUuid();
        this->outputNeuronUuid = strongOutput->getUuid();

        if (strongInput == strongOutput)
        {
            strongInput->selfConnection = this->shared_from_this();
            return;
        }
        
        // reference all the connections and traces
        strongInput->outgoingConnections[this->getUuid()] = this->shared_from_this();
        strongInput->neighbours[strongOutput->getUuid()] = strongOutput;
        
        strongOutput->incomingConnections[this->getUuid()] = this->shared_from_this();
        strongOutput->eligibility[this->getUuid()] = 0.0;
        
        for (auto &extendedTrace : strongOutput->extended)
        {
            extendedTrace.second[this->getUuid()] = 0.0;
        }
    }
    
    // =============================================================================
    // Serialization
    //
    
    inline void Neuron::Connection::deserialize(SerializationContext::Ptr context)
    {
        this->uuid = context->getStringProperty(Keys::Core::Uuid);
        this->inputNeuronUuid = context->getStringProperty(Keys::Core::InputNeuronUuid);
        this->gateNeuronUuid = context->getStringProperty(Keys::Core::GateNeuronUuid);
        this->outputNeuronUuid = context->getStringProperty(Keys::Core::OutputNeuronUuid);
    }
    
    inline void Neuron::Connection::serialize(SerializationContext::Ptr context) const
    {
        context->setStringProperty(this->uuid, Keys::Core::Uuid);
        context->setStringProperty(this->inputNeuronUuid, Keys::Core::InputNeuronUuid);
        context->setStringProperty(this->gateNeuronUuid, Keys::Core::GateNeuronUuid);
        context->setStringProperty(this->outputNeuronUuid, Keys::Core::OutputNeuronUuid);
    }
}

#endif  // TINYRNN_NEURON_H_INCLUDED
