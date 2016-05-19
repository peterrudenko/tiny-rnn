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

#ifndef TINYRNN_HARDCODEDNEURON_H_INCLUDED
#define TINYRNN_HARDCODEDNEURON_H_INCLUDED

#include "Common.h"
#include "HardcodedTrainingContext.h"
#include "Id.h"
#include "Neuron.h"

namespace TinyRNN
{
    class KernelSentence final
    {
    public:
        
        KernelSentence() = default;
        
        friend KernelSentence &operator << (KernelSentence &i, Index index);
        friend KernelSentence &operator << (KernelSentence &i, float value);
        friend KernelSentence &operator << (KernelSentence &i, const std::string &operations);
        friend void operator << (KernelSentence &i, std::ostream&(*f)(std::ostream&));
        
        size_t getSize() const noexcept;
        std::string build() const;
        
    private:
        
        std::string expressionBuilder;              // i.e. one line like "x[123] += x[0] * (x[183] * x[342]);"
        std::vector<std::string> expressions;       // an array of lines for this neuron
        
        TINYRNN_DISALLOW_COPY_AND_ASSIGN(KernelSentence);
    };
    
    // TODO(peterrudenko): rename to UnfoldedNeuron or so
    class HardcodedNeuron final
    {
    public:
        
        using Ptr = std::shared_ptr<HardcodedNeuron>;
        using Vector = std::vector<HardcodedNeuron::Ptr>;
        
    public:
        
        HardcodedNeuron() = default;
        
        static HardcodedNeuron::Ptr buildFrom(HardcodedTrainingContext::Ptr context,
                                              Neuron::Ptr target,
                                              bool asInput,
                                              bool asOutput,
                                              bool asConst);
        
        const KernelSentence &getFeedChunk() const noexcept;
        const KernelSentence &getTraceChunk() const noexcept;
        const KernelSentence &getTrainChunk() const noexcept;
        
    private:
        
        KernelSentence feedProgram;
        KernelSentence traceProgram;
        KernelSentence trainProgram;
        
        TINYRNN_DISALLOW_COPY_AND_ASSIGN(HardcodedNeuron);
    };
    
    //===------------------------------------------------------------------===//
    // KernelSentence implementation
    //===------------------------------------------------------------------===//
    
    inline size_t KernelSentence::getSize() const noexcept
    {
        return this->expressions.size();
    }
    
    inline std::string KernelSentence::build() const
    {
        std::string result;
        
        for (const auto &expression : this->expressions)
        {
            result = result + expression;
        }
        
        return result;
    }
    
    inline KernelSentence &operator << (KernelSentence &i, Index index)
    {
        i.expressionBuilder += "x[" + std::to_string(index) + "]";
        return i;
    }
    
    inline KernelSentence &operator << (KernelSentence &i, float value)
    {
        i.expressionBuilder += std::to_string(value);
        return i;
    }
    
    inline KernelSentence &operator << (KernelSentence &i, const std::string &operations)
    {
        i.expressionBuilder += operations;
        return i;
    }
    
    inline void operator << (KernelSentence &i, std::ostream&(*f)(std::ostream&))
    {
        // What a mess, I just wanted to have std::endl to finish the current line..
        if (f == (std::ostream&(*)(std::ostream&)) &std::endl)
        {
            if (i.expressionBuilder.back() != ';')
            {
                i.expressionBuilder += ";";
            }
            
            i.expressionBuilder += "\n";
            i.expressions.push_back(i.expressionBuilder);
            i.expressionBuilder.clear();
        }
    }
    
    //===------------------------------------------------------------------===//
    // HardcodedNeuron implementation
    //===------------------------------------------------------------------===//
    
    inline HardcodedNeuron::Ptr HardcodedNeuron::buildFrom(HardcodedTrainingContext::Ptr context,
                                                           Neuron::Ptr target,
                                                           bool asInput,
                                                           bool asOutput,
                                                           bool asConst)
    {
        HardcodedNeuron::Ptr hardcoded(new HardcodedNeuron());
        
        auto targetData = target->getTrainingData();
        
        const Index rateVar =
        context->allocateOrReuseVariable(0, {Keys::Mapping::Rate});
        
        context->registerRateVariable(rateVar);
        
        const Index activationVar =
        context->allocateOrReuseVariable(targetData->activation,
                                         {target->getUuid(), Keys::Mapping::Activation});
        
        const Index derivativeVar =
        context->allocateOrReuseVariable(targetData->derivative,
                                         {target->getUuid(), Keys::Mapping::Derivative});
        
        if (asInput)
        {
            context->registerInputVariable(activationVar);
        }
        else
        {
            const Index biasVar =
            context->allocateOrReuseVariable(targetData->bias,
                                             {target->getUuid(), Keys::Mapping::Bias});
            
            const Index stateVar =
            context->allocateOrReuseVariable(targetData->state,
                                             {target->getUuid(), Keys::Mapping::State});
            
            const Index oldStateVar =
            context->allocateOrReuseVariable(targetData->oldState,
                                             {target->getUuid(), Keys::Mapping::OldState});
            
            Index selfConnectionGainVar = 0;
            Index selfConnectionWeightVar = 0;
            
            if (target->isSelfConnected())
            {
                const auto selfConnectionData = target->selfConnection->getTrainingData();
                
                selfConnectionWeightVar =
                context->allocateOrReuseVariable(selfConnectionData->weight,
                                                 {target->selfConnection->getUuid(), Keys::Mapping::Weight});
                
                const bool selfConnectionHasGate = (target->selfConnection->getGateNeuron() != nullptr);
                if (selfConnectionHasGate)
                {
                    selfConnectionGainVar =
                    context->allocateOrReuseVariable(selfConnectionData->gain,
                                                     {target->selfConnection->getUuid(), Keys::Mapping::Gain});
                    
                }
            }
            
            hardcoded->feedProgram << oldStateVar << " = " << stateVar << std::endl;
            
            // eq. 15
            if (target->isSelfConnected())
            {
                const auto selfConnectionData = target->selfConnection->getTrainingData();
                const Index selfWeightVar =
                context->allocateOrReuseVariable(selfConnectionData->weight,
                                                 {target->selfConnection->getUuid(), Keys::Mapping::Weight});
                
                if (target->selfConnection->getGateNeuron() != nullptr)
                {
                    const Index selfGainVar =
                    context->allocateOrReuseVariable(selfConnectionData->gain,
                                                     {target->selfConnection->getUuid(), Keys::Mapping::Gain});
                    
                    hardcoded->feedProgram << stateVar << " = " << selfGainVar << " * " << selfWeightVar << " * " << stateVar << " + " << biasVar << std::endl;
                }
                else
                {
                    hardcoded->feedProgram << stateVar << " = " << selfWeightVar << " * " << stateVar << " + " << biasVar << std::endl;
                }
            }
            else
            {
                hardcoded->feedProgram << stateVar << " = " << biasVar << std::endl;
            }
            
            for (auto &i : target->incomingConnections)
            {
                const Neuron::Connection::Ptr inputConnection = i.second;
                const Neuron::Ptr inputNeuron = inputConnection->getInputNeuron();
                const auto inputConnectionData = inputConnection->getTrainingData();
                const auto inputNeuronData = inputNeuron->getTrainingData();
                
                const Index inputActivationVar =
                context->allocateOrReuseVariable(inputNeuronData->activation,
                                                 {inputNeuron->getUuid(), Keys::Mapping::Activation});
                
                const Index inputWeightVar =
                context->allocateOrReuseVariable(inputConnectionData->weight,
                                                 {inputConnection->getUuid(), Keys::Mapping::Weight});
                
                if (inputConnection->getGateNeuron() != nullptr)
                {
                    const Index inputGainVar =
                    context->allocateOrReuseVariable(inputConnectionData->gain,
                                                     {inputConnection->getUuid(), Keys::Mapping::Gain});
                    
                    hardcoded->feedProgram << stateVar << " += " << inputActivationVar << " * " << inputWeightVar << " * " << inputGainVar << std::endl;
                }
                else
                {
                    hardcoded->feedProgram << stateVar << " += " << inputActivationVar << " * " << inputWeightVar << std::endl;
                }
            }
            
            // eq. 16
            hardcoded->feedProgram << activationVar << " = (1.0 / (1.0 + exp(-" << stateVar << ")))" << std::endl;
            
            // f'(s)
            hardcoded->feedProgram << derivativeVar << " = " << activationVar << " * (1.0 - " << activationVar << ")" << std::endl;
            
            if (! asConst)
            {
                // Calculate extended elegibility traces in advance
                Neuron::EligibilityMap influences;
                
                for (auto &i : target->extended)
                {
                    // extended elegibility trace
                    const Id neuronId = i.first;
                    const Value influence = influences[neuronId];
                    
                    Neuron::Ptr neighbour = target->neighbours[i.first];
                    const Index influenceVar =
                    context->allocateOrReuseVariable(influence,
                                                     {neighbour->getUuid(), Keys::Mapping::Influence});
                    
                    auto neighbourData = neighbour->getTrainingData();
                    const Index neighbourOldStateVar =
                    context->allocateOrReuseVariable(neighbourData->oldState,
                                                     {neighbour->getUuid(), Keys::Mapping::OldState});
                    
                    bool influenceWasInitialized = false;
                    
                    // if gated neuron's selfconnection is gated by this unit, the influence keeps track of the neuron's old state
                    if (Neuron::Connection::Ptr neighbourSelfconnection = neighbour->getSelfConnection())
                    {
                        if (neighbourSelfconnection->getGateNeuron() == target)
                        {
                            hardcoded->traceProgram << influenceVar << " = " << neighbourOldStateVar << std::endl;
                            influenceWasInitialized = true;
                        }
                    }
                    
                    // index runs over all the incoming connections to the gated neuron that are gated by this unit
                    for (auto &incoming : target->influences[neighbour->getUuid()])
                    { // captures the effect that has an input connection to this unit, on a neuron that is gated by this unit
                        const Neuron::Connection::Ptr inputConnection = incoming.second;
                        const Neuron::Ptr inputNeuron = inputConnection->getInputNeuron();
                        const auto inputConnectionData = inputConnection->getTrainingData();
                        const auto inputNeuronData = inputNeuron->getTrainingData();
                        
                        const Index incomingWeightVar =
                        context->allocateOrReuseVariable(inputConnectionData->weight,
                                                         {inputConnection->getUuid(), Keys::Mapping::Weight});
                        
                        const Index incomingActivationVar =
                        context->allocateOrReuseVariable(inputNeuronData->activation,
                                                         {inputNeuron->getUuid(), Keys::Mapping::Activation});
                        
                        if (influenceWasInitialized)
                        {
                            hardcoded->traceProgram << influenceVar << " += " << incomingWeightVar << " * " << incomingActivationVar << std::endl;
                        }
                        else
                        {
                            hardcoded->traceProgram << influenceVar << " = " << incomingWeightVar << " * " << incomingActivationVar << std::endl;
                            influenceWasInitialized = true;
                        }
                    }
                }
                
                for (auto &i : target->incomingConnections)
                {
                    const Neuron::Connection::Ptr inputConnection = i.second;
                    const Neuron::Ptr inputNeuron = inputConnection->getInputNeuron();
                    const bool inputConnectionHasGate = (inputConnection->getGateNeuron() != nullptr);
                    
                    // elegibility trace - Eq. 17
                    const auto inputConnectionData = inputConnection->getTrainingData();
                    const auto inputNeuronData = inputNeuron->getTrainingData();
                    
                    Index inputGainVar = 0;
                    if (inputConnectionHasGate)
                    {
                        inputGainVar =
                        context->allocateOrReuseVariable(inputConnectionData->gain,
                                                         {inputConnection->getUuid(), Keys::Mapping::Gain});
                    }
                    
                    const Index inputActivationVar =
                    context->allocateOrReuseVariable(inputNeuronData->activation,
                                                     {inputNeuron->getUuid(), Keys::Mapping::Activation});
                    
                    const Index eligibilityVar =
                    context->allocateOrReuseVariable(target->eligibility[inputConnection->getUuid()],
                                                     {target->getUuid(), inputConnection->getUuid(), Keys::Mapping::Eligibility});
                    
                    if (target->isSelfConnected())
                    {
                        const auto selfConnectionData = target->selfConnection->getTrainingData();
                        const bool selfConnectionHasGate = (target->selfConnection->getGateNeuron() != nullptr);
                        
                        if (selfConnectionHasGate)
                        {
                            if (inputConnectionHasGate)
                            {
                                hardcoded->traceProgram << eligibilityVar << " = " << selfConnectionGainVar << " * " << selfConnectionWeightVar << " * " << eligibilityVar << " + " << inputGainVar << " * " << inputActivationVar << std::endl;
                            }
                            else
                            {
                                hardcoded->traceProgram << eligibilityVar << " = " << selfConnectionGainVar << " * " << selfConnectionWeightVar << " * " << eligibilityVar << " + " << inputActivationVar << std::endl;
                            }
                        }
                        else
                        {
                            if (inputConnectionHasGate)
                            {
                                hardcoded->traceProgram << eligibilityVar << " = " << selfConnectionWeightVar << " * " << eligibilityVar << " + " << inputGainVar << " * " << inputActivationVar << std::endl;
                                
                            }
                            else
                            {
                                hardcoded->traceProgram << eligibilityVar << " = " << selfConnectionWeightVar << " * " << eligibilityVar << " + " << inputActivationVar << std::endl;
                            }
                        }
                    }
                    else
                    {
                        if (inputConnectionHasGate)
                        {
                            hardcoded->traceProgram << eligibilityVar << " = " << inputGainVar << " * " << inputActivationVar << std::endl;
                        }
                        else
                        {
                            hardcoded->traceProgram << eligibilityVar << " = " << inputActivationVar << std::endl;
                        }
                    }
                    
                    for (auto &i : target->extended)
                    {
                        // extended elegibility trace
                        const Id neighbourNeuronUuid = i.first;
                        const Value influence = influences[neighbourNeuronUuid];
                        
                        Neuron::EligibilityMap &xtrace = i.second;
                        Neuron::Ptr neighbour = target->neighbours[neighbourNeuronUuid];
                        
                        const auto neighbourData = neighbour->getTrainingData();
                        
                        const Index influenceVar =
                        context->allocateOrReuseVariable(influence,
                                                         {neighbour->getUuid(), Keys::Mapping::Influence});
                        
                        const Index eligibilityVar =
                        context->allocateOrReuseVariable(target->eligibility[inputConnection->getUuid()],
                                                         {target->getUuid(), inputConnection->getUuid(), Keys::Mapping::Eligibility});
                        
                        const Index extendedTraceVar =
                        context->allocateOrReuseVariable(xtrace[inputConnection->getUuid()],
                                                         {target->getUuid(), neighbourNeuronUuid, inputConnection->getUuid(), Keys::Mapping::ExtendedTrace});
                        
                        if (Neuron::Connection::Ptr neighbourSelfConnection = neighbour->getSelfConnection())
                        {
                            const auto neighbourSelfConnectionData = neighbourSelfConnection->getTrainingData();
                            
                            if (neighbourSelfConnection->getGateNeuron() != nullptr)
                            {
                                hardcoded->traceProgram << extendedTraceVar << " = " << selfConnectionGainVar << " * " << selfConnectionWeightVar << " * " << extendedTraceVar << " + " << derivativeVar << " * " << eligibilityVar << " * " << influenceVar << std::endl;
                            }
                            else
                            {
                                hardcoded->traceProgram << extendedTraceVar << " = " << selfConnectionWeightVar << " * " << extendedTraceVar << " + " << derivativeVar << " * " << eligibilityVar << " * " << influenceVar << std::endl;
                            }
                        }
                        else
                        {
                            hardcoded->traceProgram << extendedTraceVar << " = " << derivativeVar << " * " << eligibilityVar << " * " << influenceVar << std::endl;
                        }
                    }
                }
            }
            
            // update gated connection's gains
            for (auto &i : target->gatedConnections)
            {
                const Neuron::Connection::Ptr gatedConnection = i.second;
                const auto gatedConnectionData = gatedConnection->getTrainingData();
                
                const Index gatedConnectionGainVar =
                context->allocateOrReuseVariable(gatedConnectionData->gain,
                                                 {gatedConnection->getUuid(), Keys::Mapping::Gain});
                
                hardcoded->feedProgram << gatedConnectionGainVar << " = " << activationVar << std::endl;
            }
        }
        
        // Done with feed program, now fix the train program:
        
        if (!asInput &&
            !asConst)
        {
            const Index responsibilityVar =
            context->allocateOrReuseVariable(targetData->errorResponsibility,
                                             {target->getUuid(), Keys::Mapping::ErrorResponsibility});
            
            const bool noOutgoingConnections = target->outgoingConnections.empty();
            const bool noGates = target->gatedConnections.empty();
            
            if (asOutput)
            {
                const Index myTargetVar =
                context->allocateOrReuseVariable(0.0,
                                                 {target->getUuid(), Keys::Mapping::Target});
                
                context->registerTargetVariable(myTargetVar);
                context->registerOutputVariable(activationVar);
                
                hardcoded->trainProgram << responsibilityVar << " = " << myTargetVar << " - " << activationVar << std::endl;
                
                for (auto &i : target->incomingConnections)
                {
                    const Neuron::Connection::Ptr inputConnection = i.second;
                    auto inputConnectionData = inputConnection->getTrainingData();
                    
                    const Index eligibilityVar =
                    context->allocateOrReuseVariable(target->eligibility[inputConnection->getUuid()],
                                                     {target->getUuid(), inputConnection->getUuid(), Keys::Mapping::Eligibility});
                    
                    const Index inputWeightVar =
                    context->allocateOrReuseVariable(inputConnectionData->weight,
                                                     {inputConnection->getUuid(), Keys::Mapping::Weight});
                    
                    hardcoded->trainProgram << inputWeightVar << " += " << rateVar << " * (" << responsibilityVar << " * " << eligibilityVar << ")" << std::endl;
                }
            }
            else
            {
                if (!noOutgoingConnections && !noGates)
                {
                    const Index errorAccumulatorVar =
                    context->allocateOrReuseVariable(0.0,
                                                     {Keys::Mapping::ErrorAccumulator});
                    
                    // error responsibilities from all the connections projected from this neuron
                    for (auto &i : target->outgoingConnections)
                    {
                        const Neuron::Connection::Ptr outputConnection = i.second;
                        const Neuron::Ptr outputNeuron = outputConnection->getOutputNeuron();
                        const auto outputConnectionData = outputConnection->getTrainingData();
                        const auto outputNeuronData = outputNeuron->getTrainingData();
                        
                        const Index outputWeightVar =
                        context->allocateOrReuseVariable(outputConnectionData->weight,
                                                         {outputConnection->getUuid(), Keys::Mapping::Weight});
                        
                        const Index outputResponsibilityVar =
                        context->allocateOrReuseVariable(outputNeuronData->errorResponsibility,
                                                         {outputNeuron->getUuid(), Keys::Mapping::ErrorResponsibility});
                        
                        if (outputConnection->getGateNeuron() != nullptr)
                        {
                            const Index outputGainVar =
                            context->allocateOrReuseVariable(outputConnectionData->gain,
                                                             {outputConnection->getUuid(), Keys::Mapping::Gain});
                            
                            hardcoded->trainProgram << errorAccumulatorVar << " += " << outputResponsibilityVar << " * " << outputGainVar << " * " << outputWeightVar << std::endl;
                        }
                        else
                        {
                            hardcoded->trainProgram << errorAccumulatorVar << " += " << outputResponsibilityVar << " * " << outputWeightVar << std::endl;
                        }
                    }
                    
                    const Index projectedErrorVar =
                    context->allocateOrReuseVariable(targetData->projectedActivity,
                                                     {target->getUuid(), Keys::Mapping::ProjectedActivity});
                    
                    // projected error responsibility
                    hardcoded->trainProgram << projectedErrorVar << " = " << derivativeVar << " * " << errorAccumulatorVar << std::endl;
                    hardcoded->trainProgram << errorAccumulatorVar << " = 0" << std::endl;
                    
                    // error responsibilities from all the connections gated by this neuron
                    for (auto &i : target->extended)
                    {
                        const Id gatedNeuronId = i.first;
                        const Neuron::Ptr gatedNeuron = target->neighbours[gatedNeuronId];
                        const auto gatedNeuronData = gatedNeuron->getTrainingData();
                        
                        const Index influenceTempVar =
                        context->allocateOrReuseVariable(0.0,
                                                         {Keys::Mapping::Influence});
                        
                        const Index gatedNeuronOldStateVar =
                        context->allocateOrReuseVariable(gatedNeuronData->oldState,
                                                         {gatedNeuron->getUuid(), Keys::Mapping::OldState});
                        
                        // if gated neuron's selfconnection is gated by this neuron
                        if (auto gatedNeuronSelfConnection = gatedNeuron->getSelfConnection())
                        {
                            if (gatedNeuronSelfConnection->getGateNeuron() == target)
                            {
                                hardcoded->trainProgram << influenceTempVar << " = " << gatedNeuronOldStateVar << std::endl;
                            }
                            else
                            {
                                hardcoded->trainProgram << influenceTempVar << " = 0" << std::endl;
                            }
                        }
                        
                        if (! asConst)
                        {
                            // index runs over all the connections to the gated neuron that are gated by this neuron
                            for (auto &i : target->influences[gatedNeuronId])
                            { // captures the effect that the input connection of this neuron have, on a neuron which its input/s is/are gated by this neuron
                                const Neuron::Connection::Ptr inputConnection = i.second;
                                const Neuron::Ptr inputNeuron = inputConnection->getInputNeuron();
                                const auto inputConnectionData = inputConnection->getTrainingData();
                                const auto inputNeuronData = inputNeuron->getTrainingData();
                                
                                const Index inputActivationVar =
                                context->allocateOrReuseVariable(inputNeuronData->activation,
                                                                 {inputNeuron->getUuid(), Keys::Mapping::Activation});
                                
                                const Index inputWeightVar =
                                context->allocateOrReuseVariable(inputConnectionData->weight,
                                                                 {inputConnection->getUuid(), Keys::Mapping::Weight});
                                
                                hardcoded->trainProgram << influenceTempVar << " += " << inputWeightVar << " * " << inputActivationVar << std::endl;
                            }
                        }
                        
                        const Index gatedResponsibilityVar =
                        context->allocateOrReuseVariable(gatedNeuronData->errorResponsibility,
                                                         {gatedNeuron->getUuid(), Keys::Mapping::ErrorResponsibility});
                        
                        // eq. 22
                        hardcoded->trainProgram << errorAccumulatorVar << " += " << gatedResponsibilityVar << " * " << influenceTempVar << std::endl;
                    }
                    
                    const Index gatedErrorVar =
                    context->allocateOrReuseVariable(targetData->gatingActivity,
                                                     {target->getUuid(), Keys::Mapping::GatingActivity});
                    
                    // gated error responsibility
                    hardcoded->trainProgram << gatedErrorVar << " = " << derivativeVar << " * " << errorAccumulatorVar << std::endl;
                    
                    // error responsibility - Eq. 23
                    hardcoded->trainProgram << responsibilityVar << " = " << projectedErrorVar << " + " << gatedErrorVar << std::endl;
                    
                    // adjust all the neuron's incoming connections
                    for (auto &i : target->incomingConnections)
                    {
                        const Id inputConnectionUuid = i.first;
                        const Neuron::Connection::Ptr inputConnection = i.second;
                        
                        const Index gradientTempVar =
                        context->allocateOrReuseVariable(0.0,
                                                         {Keys::Mapping::Gradient});
                        
                        const Index eligibilityVar =
                        context->allocateOrReuseVariable(target->eligibility[inputConnection->getUuid()],
                                                         {target->getUuid(), inputConnection->getUuid(), Keys::Mapping::Eligibility});
                        
                        // Eq. 24
                        hardcoded->trainProgram << gradientTempVar << " = " << projectedErrorVar << " * " << eligibilityVar << std::endl;
                        
                        for (auto &ext : target->extended)
                        {
                            // extended elegibility trace
                            const Id neighbourNeuronId = ext.first;
                            Neuron::EligibilityMap &xtrace = ext.second;
                            Neuron::Ptr neighbour = target->neighbours[neighbourNeuronId];
                            const auto neighbourData = neighbour->getTrainingData();
                            
                            const Index neighbourResponsibilityVar =
                            context->allocateOrReuseVariable(neighbourData->errorResponsibility,
                                                             {neighbourNeuronId, Keys::Mapping::ErrorResponsibility});
                            
                            const Index extendedTraceVar =
                            context->allocateOrReuseVariable(xtrace[inputConnection->getUuid()],
                                                             {target->getUuid(), neighbourNeuronId, inputConnectionUuid, Keys::Mapping::ExtendedTrace});
                            
                            hardcoded->trainProgram << gradientTempVar << " += " << neighbourResponsibilityVar << " * " << extendedTraceVar << std::endl;
                        }
                        
                        // adjust weights - aka learn
                        auto inputConnectionData = inputConnection->getTrainingData();
                        
                        const Index inputWeightVar =
                        context->allocateOrReuseVariable(inputConnectionData->weight,
                                                         {inputConnection->getUuid(), Keys::Mapping::Weight});
                        
                        hardcoded->trainProgram << inputWeightVar << " += " << rateVar << " * " << gradientTempVar << std::endl;
                    }
                }
                else if (noGates)
                {
                    hardcoded->trainProgram << responsibilityVar << " = 0" << std::endl;
                    
                    // error responsibilities from all the connections projected from this neuron
                    for (auto &i : target->outgoingConnections)
                    {
                        const Neuron::Connection::Ptr outputConnection = i.second;
                        const Neuron::Ptr outputNeuron = outputConnection->getOutputNeuron();
                        const auto outputConnectionData = outputConnection->getTrainingData();
                        const auto outputNeuronData = outputNeuron->getTrainingData();
                        
                        const Index outputWeightVar =
                        context->allocateOrReuseVariable(outputConnectionData->weight,
                                                         {outputConnection->getUuid(), Keys::Mapping::Weight});
                        
                        const Index outputResponsibilityVar =
                        context->allocateOrReuseVariable(outputNeuronData->errorResponsibility,
                                                         {outputNeuron->getUuid(), Keys::Mapping::ErrorResponsibility});
                        
                        if (outputConnection->getGateNeuron() != nullptr)
                        {
                            const Index outputGainVar =
                            context->allocateOrReuseVariable(outputConnectionData->gain,
                                                             {outputConnection->getUuid(), Keys::Mapping::Gain});
                            
                            hardcoded->trainProgram << responsibilityVar << " += " << outputResponsibilityVar << " * " << outputGainVar << " * " << outputWeightVar << std::endl;
                        }
                        else
                        {
                            hardcoded->trainProgram << responsibilityVar << " += " << outputResponsibilityVar << " * " << outputWeightVar << std::endl;
                        }
                    }
                    
                    hardcoded->trainProgram << responsibilityVar << " *= " << derivativeVar << std::endl;
                    
                    for (auto &i : target->incomingConnections)
                    {
                        const Neuron::Connection::Ptr inputConnection = i.second;
                        auto inputConnectionData = inputConnection->getTrainingData();
                        
                        const Index eligibilityVar =
                        context->allocateOrReuseVariable(target->eligibility[inputConnection->getUuid()],
                                                         {target->getUuid(), inputConnection->getUuid(), Keys::Mapping::Eligibility});
                        
                        const Index inputWeightVar =
                        context->allocateOrReuseVariable(inputConnectionData->weight,
                                                         {inputConnection->getUuid(), Keys::Mapping::Weight});
                        
                        // learn
                        hardcoded->trainProgram << inputWeightVar << " += " << rateVar << " * (" << responsibilityVar << " * " << eligibilityVar << ")" << std::endl;
                    }
                }
                else if (noOutgoingConnections)
                {
                    hardcoded->trainProgram << responsibilityVar << " = 0" << std::endl;
                    
                    // error responsibilities from all the connections gated by this neuron
                    for (auto &i : target->extended)
                    {
                        const Id gatedNeuronId = i.first;
                        const Neuron::Ptr gatedNeuron = target->neighbours[gatedNeuronId];
                        const auto gatedNeuronData = gatedNeuron->getTrainingData();
                        
                        const Index influenceTempVar =
                        context->allocateOrReuseVariable(0.0,
                                                         {Keys::Mapping::Influence});
                        
                        const Index gatedNeuronOldStateVar =
                        context->allocateOrReuseVariable(gatedNeuronData->oldState,
                                                         {gatedNeuron->getUuid(), Keys::Mapping::OldState});
                        
                        // if gated neuron's selfconnection is gated by this neuron
                        if (auto gatedNeuronSelfConnection = gatedNeuron->getSelfConnection())
                        {
                            if (gatedNeuronSelfConnection->getGateNeuron() == target)
                            {
                                hardcoded->trainProgram << influenceTempVar << " = " << gatedNeuronOldStateVar << std::endl;
                            }
                            else
                            {
                                hardcoded->trainProgram << influenceTempVar << " = 0" << std::endl;
                            }
                        }
                        
                        // index runs over all the connections to the gated neuron that are gated by this neuron
                        for (auto &i : target->influences[gatedNeuronId])
                        { // captures the effect that the input connection of this neuron have, on a neuron which its input/s is/are gated by this neuron
                            const Neuron::Connection::Ptr inputConnection = i.second;
                            const Neuron::Ptr inputNeuron = inputConnection->getInputNeuron();
                            const auto inputConnectionData = inputConnection->getTrainingData();
                            const auto inputNeuronData = inputNeuron->getTrainingData();
                            
                            const Index inputActivationVar =
                            context->allocateOrReuseVariable(inputNeuronData->activation,
                                                             {inputNeuron->getUuid(), Keys::Mapping::Activation});
                            
                            const Index inputWeightVar =
                            context->allocateOrReuseVariable(inputConnectionData->weight,
                                                             {inputConnection->getUuid(), Keys::Mapping::Weight});
                            
                            hardcoded->trainProgram << influenceTempVar << " += " << inputWeightVar << " * " << inputActivationVar << std::endl;
                        }
                        
                        const Index gatedResponsibilityVar =
                        context->allocateOrReuseVariable(gatedNeuronData->errorResponsibility,
                                                         {gatedNeuron->getUuid(), Keys::Mapping::ErrorResponsibility});
                        
                        // eq. 22
                        hardcoded->trainProgram << responsibilityVar << " += " << gatedResponsibilityVar << " * " << influenceTempVar << std::endl;
                    }
                    
                    hardcoded->trainProgram << responsibilityVar << " *= " << derivativeVar << std::endl;
                    
                    // adjust all the neuron's incoming connections
                    for (auto &i : target->incomingConnections)
                    {
                        const Id inputConnectionUuid = i.first;
                        const Neuron::Connection::Ptr inputConnection = i.second;
                        
                        const Index gradientTempVar =
                        context->allocateOrReuseVariable(0.0,
                                                         {Keys::Mapping::Gradient});
                        
                        hardcoded->trainProgram << gradientTempVar << " = 0" << std::endl;
                        
                        for (auto &ext : target->extended)
                        {
                            // extended elegibility trace
                            const Id neighbourNeuronId = ext.first;
                            Neuron::EligibilityMap &xtrace = ext.second;
                            Neuron::Ptr neighbour = target->neighbours[neighbourNeuronId];
                            const auto neighbourData = neighbour->getTrainingData();
                            
                            const Index neighbourResponsibilityVar =
                            context->allocateOrReuseVariable(neighbourData->errorResponsibility,
                                                             {neighbourNeuronId, Keys::Mapping::ErrorResponsibility});
                            
                            const Index extendedTraceVar =
                            context->allocateOrReuseVariable(xtrace[inputConnection->getUuid()],
                                                             {target->getUuid(), neighbourNeuronId, inputConnectionUuid, Keys::Mapping::ExtendedTrace});
                            
                            hardcoded->trainProgram << gradientTempVar << " += " << neighbourResponsibilityVar << " * " << extendedTraceVar << std::endl;
                        }
                        
                        // adjust weights - aka learn
                        auto inputConnectionData = inputConnection->getTrainingData();
                        
                        const Index inputWeightVar =
                        context->allocateOrReuseVariable(inputConnectionData->weight,
                                                         {inputConnection->getUuid(), Keys::Mapping::Weight});
                        
                        hardcoded->trainProgram << inputWeightVar << " += " << rateVar << " * " << gradientTempVar << std::endl;
                    }
                }
            }
            
            // adjust bias
            const Index biasVar =
            context->allocateOrReuseVariable(targetData->bias,
                                             {target->getUuid(), Keys::Mapping::Bias});
            
            hardcoded->trainProgram << biasVar << " += " << rateVar << " * " << responsibilityVar << std::endl;
        }
        
        return hardcoded;
    }
    
    inline const KernelSentence &HardcodedNeuron::getFeedChunk() const noexcept
    {
        return this->feedProgram;
    }
    
    inline const KernelSentence &HardcodedNeuron::getTraceChunk() const noexcept
    {
        return this->traceProgram;
    }
    
    inline const KernelSentence &HardcodedNeuron::getTrainChunk() const noexcept
    {
        return this->trainProgram;
    }
} // namespace TinyRNN

#endif // TINYRNN_HARDCODEDNEURON_H_INCLUDED
