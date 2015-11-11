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

#ifndef HARDCODEDNEURON_H_INCLUDED
#define HARDCODEDNEURON_H_INCLUDED

#if TINYRNN_OPENCL_ACCELERATION

#include "Common.h"
#include "Neuron.h"
#include "HardcodedTrainingContext.h"
#include "SerializedObject.h"

namespace TinyRNN
{
    class HardcodedNeuron final
    {
    public:
        
        using Ptr = std::shared_ptr<HardcodedNeuron>;
        using Array = std::vector<HardcodedNeuron::Ptr>;
        
    public:
        
        HardcodedNeuron();
        
        static HardcodedNeuron::Ptr buildFrom(HardcodedTrainingContext::Ptr context,
                                              Neuron::Ptr target,
                                              bool asInput,
                                              bool asOutput);
        
        static void restoreNeuronState(Neuron::Ptr targetNeuron,
                                       HardcodedTrainingContext::Ptr context);
        
        const KernelSentence &getFeedChunk() const noexcept;
        const KernelSentence &getTraceChunk() const noexcept;
        const KernelSentence &getTrainChunk() const noexcept;
        
    private:
        
        KernelSentence feedProgram;
        KernelSentence traceProgram;
        KernelSentence trainProgram;
        
        TINYRNN_DISALLOW_COPY_AND_ASSIGN(HardcodedNeuron);
    };
    
    // =============================================================================
    // HardcodedNeuron implementation
    //
    
    inline HardcodedNeuron::HardcodedNeuron() {}

    inline HardcodedNeuron::Ptr HardcodedNeuron::buildFrom(HardcodedTrainingContext::Ptr context, Neuron::Ptr target, bool asInput, bool asOutput)
    {
        HardcodedNeuron::Ptr hardcoded(new HardcodedNeuron());
        
        auto targetData = target->getTrainingData();
        
        const size_t rateVar =
        context->allocateOrReuseVariable(0, {Keys::Core::Rate});
        
        context->registerRateVariable(rateVar);
        
        const size_t activationVar =
        context->allocateOrReuseVariable(targetData->activation,
        {target->getUuid(), Keys::Core::Activation});
        
        const size_t derivativeVar =
        context->allocateOrReuseVariable(targetData->derivative,
        {target->getUuid(), Keys::Core::Derivative});
        
        if (asInput)
        {
            context->registerInputVariable(activationVar);
        }
        else
        {
            const size_t biasVar =
            context->allocateOrReuseVariable(targetData->bias,
            {target->getUuid(), Keys::Core::Bias});
            
            const size_t stateVar =
            context->allocateOrReuseVariable(targetData->state,
            {target->getUuid(), Keys::Core::State});
            
            const size_t oldStateVar =
            context->allocateOrReuseVariable(targetData->oldState,
            {target->getUuid(), Keys::Core::OldState});
            
            size_t selfConnectionGainVar = 0;
            size_t selfConnectionWeightVar = 0;
            
            if (target->isSelfConnected())
            {
                const auto selfConnectionData = target->selfConnection->getTrainingData();
                
                selfConnectionWeightVar =
                context->allocateOrReuseVariable(selfConnectionData->weight,
                {target->selfConnection->getUuid(), Keys::Core::Weight});
                
                const bool selfConnectionHasGate = (target->selfConnection->getGateNeuron() != nullptr);
                if (selfConnectionHasGate)
                {
                    selfConnectionGainVar =
                    context->allocateOrReuseVariable(selfConnectionData->gain,
                    {target->selfConnection->getUuid(), Keys::Core::Gain});
                    
                }
            }
            
            hardcoded->feedProgram << oldStateVar << " = " << stateVar << std::endl;
            
            // eq. 15
            if (target->isSelfConnected())
            {
                const auto selfConnectionData = target->selfConnection->getTrainingData();
                const size_t selfWeightVar =
                context->allocateOrReuseVariable(selfConnectionData->weight,
                {target->selfConnection->getUuid(), Keys::Core::Weight});
                
                if (target->selfConnection->getGateNeuron() != nullptr)
                {
                    const size_t selfGainVar =
                    context->allocateOrReuseVariable(selfConnectionData->gain,
                    {target->selfConnection->getUuid(), Keys::Core::Gain});
                    
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
                
                const size_t inputActivationVar =
                context->allocateOrReuseVariable(inputNeuronData->activation,
                {inputNeuron->getUuid(), Keys::Core::Activation});
                
                const size_t inputWeightVar =
                context->allocateOrReuseVariable(inputConnectionData->weight,
                {inputConnection->getUuid(), Keys::Core::Weight});
                
                if (inputConnection->getGateNeuron() != nullptr)
                {
                    const size_t inputGainVar =
                    context->allocateOrReuseVariable(inputConnectionData->gain,
                    {inputConnection->getUuid(), Keys::Core::Gain});
                    
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
            
            // Calculate extended elegibility traces in advance
            Neuron::EligibilityMap influences;
            for (auto &i : target->extended)
            {
                // extended elegibility trace
                const std::string neuronId = i.first;
                const double influence = influences[neuronId];
                
                Neuron::Ptr neighbour = target->neighbours[i.first];
                const size_t influenceVar =
                context->allocateOrReuseVariable(influence,
                {neighbour->getUuid(), Keys::Core::Influence});
                
                auto neighbourData = neighbour->getTrainingData();
                const size_t neighbourOldStateVar =
                context->allocateOrReuseVariable(neighbourData->oldState,
                {neighbour->getUuid(), Keys::Core::OldState});
                
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
                    
                    const size_t incomingWeightVar =
                    context->allocateOrReuseVariable(inputConnectionData->weight,
                    {inputConnection->getUuid(), Keys::Core::Weight});
                    
                    const size_t incomingActivationVar =
                    context->allocateOrReuseVariable(inputNeuronData->activation,
                    {inputNeuron->getUuid(), Keys::Core::Activation});
                    
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
                const double oldElegibility = target->eligibility[inputConnection->getUuid()];
                const auto inputConnectionData = inputConnection->getTrainingData();
                const auto inputNeuronData = inputNeuron->getTrainingData();
                
                size_t inputGainVar = 0;
                if (inputConnectionHasGate)
                {
                    inputGainVar =
                    context->allocateOrReuseVariable(inputConnectionData->gain,
                    {inputConnection->getUuid(), Keys::Core::Gain});
                }
                
                const size_t inputActivationVar =
                context->allocateOrReuseVariable(inputNeuronData->activation,
                {inputNeuron->getUuid(), Keys::Core::Activation});
                
                const size_t eligibilityVar =
                context->allocateOrReuseVariable(target->eligibility[inputConnection->getUuid()],
                {target->getUuid(), inputConnection->getUuid(), Keys::Core::Eligibility});
                
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
                    const std::string neighbourNeuronUuid = i.first;
                    const double influence = influences[neighbourNeuronUuid];
                    
                    Neuron::EligibilityMap &xtrace = i.second;
                    Neuron::Ptr neighbour = target->neighbours[neighbourNeuronUuid];
                    
                    const auto neighbourData = neighbour->getTrainingData();
                    
                    const size_t influenceVar =
                    context->allocateOrReuseVariable(influence,
                    {neighbour->getUuid(), Keys::Core::Influence});
                    
                    const size_t neighbourOldStateVar =
                    context->allocateOrReuseVariable(neighbourData->oldState,
                    {neighbour->getUuid(), Keys::Core::OldState});
                    
                    const size_t eligibilityVar =
                    context->allocateOrReuseVariable(target->eligibility[inputConnection->getUuid()],
                    {target->getUuid(), inputConnection->getUuid(), Keys::Core::Eligibility});
                    
                    const size_t extendedTraceVar =
                    context->allocateOrReuseVariable(xtrace[inputConnection->getUuid()],
                    {target->getUuid(), neighbourNeuronUuid, inputConnection->getUuid(), Keys::Core::ExtendedTrace});
                    
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
            
            // update gated connection's gains
            for (auto &i : target->gatedConnections)
            {
                const Neuron::Connection::Ptr gatedConnection = i.second;
                const auto gatedConnectionData = gatedConnection->getTrainingData();
                
                const size_t gatedConnectionGainVar =
                context->allocateOrReuseVariable(gatedConnectionData->gain,
                {gatedConnection->getUuid(), Keys::Core::Gain});
                
                hardcoded->feedProgram << gatedConnectionGainVar << " = " << activationVar << std::endl;
            }
        }
        
        // Done with feed program, now fix the train program:
        
        if (!asInput)
        {
            const size_t responsibilityVar =
            context->allocateOrReuseVariable(targetData->errorResponsibility,
            {target->getUuid(), Keys::Core::ErrorResponsibility});
            
            const bool noOutgoingConnections = target->outgoingConnections.empty();
            const bool noGates = target->gatedConnections.empty();
            
            if (asOutput)
            {
                const size_t myTargetVar =
                context->allocateOrReuseVariable(0.0,
                {target->getUuid(), Keys::Core::Target});
                
                context->registerTargetVariable(myTargetVar);
                context->registerOutputVariable(activationVar);
                
                hardcoded->trainProgram << responsibilityVar << " = " << myTargetVar << " - " << activationVar << std::endl;
                
                for (auto &i : target->incomingConnections)
                {
                    const Neuron::Connection::Ptr inputConnection = i.second;
                    auto inputConnectionData = inputConnection->getTrainingData();
                    
                    const size_t eligibilityVar =
                    context->allocateOrReuseVariable(target->eligibility[inputConnection->getUuid()],
                    {target->getUuid(), inputConnection->getUuid(), Keys::Core::Eligibility});
                    
                    const size_t inputWeightVar =
                    context->allocateOrReuseVariable(inputConnectionData->weight,
                    {inputConnection->getUuid(), Keys::Core::Weight});
                    
                    hardcoded->trainProgram << inputWeightVar << " += " << rateVar << " * (" << responsibilityVar << " * " << eligibilityVar << ")" << std::endl;
                }
            }
            else
            {
                if (!noOutgoingConnections && !noGates)
                {
                    const size_t errorAccumulatorVar =
                    context->allocateOrReuseVariable(0.0,
                    {Keys::Core::ErrorAccumulator});
                    
                    // error responsibilities from all the connections projected from this neuron
                    for (auto &i : target->outgoingConnections)
                    {
                        const Neuron::Connection::Ptr outputConnection = i.second;
                        const Neuron::Ptr outputNeuron = outputConnection->getOutputNeuron();
                        const auto outputConnectionData = outputConnection->getTrainingData();
                        const auto outputNeuronData = outputNeuron->getTrainingData();
                        
                        const size_t outputWeightVar =
                        context->allocateOrReuseVariable(outputConnectionData->weight,
                        {outputConnection->getUuid(), Keys::Core::Weight});
                        
                        const size_t outputResponsibilityVar =
                        context->allocateOrReuseVariable(outputNeuronData->errorResponsibility,
                        {outputNeuron->getUuid(), Keys::Core::ErrorResponsibility});
                        
                        if (outputConnection->getGateNeuron() != nullptr)
                        {
                            const size_t outputGainVar =
                            context->allocateOrReuseVariable(outputConnectionData->gain,
                            {outputConnection->getUuid(), Keys::Core::Gain});
                            
                            hardcoded->trainProgram << errorAccumulatorVar << " += " << outputResponsibilityVar << " * " << outputGainVar << " * " << outputWeightVar << std::endl;
                        }
                        else
                        {
                            hardcoded->trainProgram << errorAccumulatorVar << " += " << outputResponsibilityVar << " * " << outputWeightVar << std::endl;
                        }
                    }
                    
                    const size_t projectedErrorVar =
                    context->allocateOrReuseVariable(targetData->projectedActivity,
                    {target->getUuid(), Keys::Core::ProjectedActivity});
                    
                    // projected error responsibility
                    hardcoded->trainProgram << projectedErrorVar << " = " << derivativeVar << " * " << errorAccumulatorVar << std::endl;
                    hardcoded->trainProgram << errorAccumulatorVar << " = 0" << std::endl;
                    
                    // error responsibilities from all the connections gated by this neuron
                    for (auto &i : target->extended)
                    {
                        const std::string gatedNeuronId = i.first;
                        const Neuron::Ptr gatedNeuron = target->neighbours[gatedNeuronId];
                        const auto gatedNeuronData = gatedNeuron->getTrainingData();
                        
                        const size_t influenceTempVar =
                        context->allocateOrReuseVariable(0.0,
                        {Keys::Core::Influence});
                        
                        const size_t gatedNeuronOldStateVar =
                        context->allocateOrReuseVariable(gatedNeuronData->oldState,
                        {gatedNeuron->getUuid(), Keys::Core::OldState});
                        
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
                            
                            const size_t inputActivationVar =
                            context->allocateOrReuseVariable(inputNeuronData->activation,
                            {inputNeuron->getUuid(), Keys::Core::Activation});
                            
                            const size_t inputWeightVar =
                            context->allocateOrReuseVariable(inputConnectionData->weight,
                            {inputConnection->getUuid(), Keys::Core::Weight});
                            
                            hardcoded->trainProgram << influenceTempVar << " += " << inputWeightVar << " * " << inputActivationVar << std::endl;
                        }
                        
                        const size_t gatedResponsibilityVar =
                        context->allocateOrReuseVariable(gatedNeuronData->errorResponsibility,
                        {gatedNeuron->getUuid(), Keys::Core::ErrorResponsibility});
                        
                        // eq. 22
                        hardcoded->trainProgram << errorAccumulatorVar << " += " << gatedResponsibilityVar << " * " << influenceTempVar << std::endl;
                    }
                    
                    const size_t gatedErrorVar =
                    context->allocateOrReuseVariable(targetData->gatingActivity,
                    {target->getUuid(), Keys::Core::GatingActivity});
                    
                    // gated error responsibility
                    hardcoded->trainProgram << gatedErrorVar << " = " << derivativeVar << " * " << errorAccumulatorVar << std::endl;
                    
                    // error responsibility - Eq. 23
                    hardcoded->trainProgram << responsibilityVar << " = " << projectedErrorVar << " + " << gatedErrorVar << std::endl;
                    
                    // adjust all the neuron's incoming connections
                    for (auto &i : target->incomingConnections)
                    {
                        const std::string inputConnectionUuid = i.first;
                        const Neuron::Connection::Ptr inputConnection = i.second;
                        
                        const size_t gradientTempVar =
                        context->allocateOrReuseVariable(0.0,
                        {Keys::Core::Gradient});
                        
                        const size_t eligibilityVar =
                        context->allocateOrReuseVariable(target->eligibility[inputConnection->getUuid()],
                        {target->getUuid(), inputConnection->getUuid(), Keys::Core::Eligibility});
                        
                        // Eq. 24
                        hardcoded->trainProgram << gradientTempVar << " = " << projectedErrorVar << " * " << eligibilityVar << std::endl;
                        
                        for (auto &ext : target->extended)
                        {
                            // extended elegibility trace
                            const std::string neighbourNeuronId = ext.first;
                            Neuron::EligibilityMap &xtrace = ext.second;
                            Neuron::Ptr neighbour = target->neighbours[neighbourNeuronId];
                            const auto neighbourData = neighbour->getTrainingData();
                            
                            const size_t neighbourResponsibilityVar =
                            context->allocateOrReuseVariable(neighbourData->errorResponsibility,
                            {neighbourNeuronId, Keys::Core::ErrorResponsibility});
                            
                            const size_t extendedTraceVar =
                            context->allocateOrReuseVariable(xtrace[inputConnection->getUuid()],
                            {target->getUuid(), neighbourNeuronId, inputConnectionUuid, Keys::Core::ExtendedTrace});
                            
                            hardcoded->trainProgram << gradientTempVar << " += " << neighbourResponsibilityVar << " * " << extendedTraceVar << std::endl;
                        }
                        
                        // adjust weights - aka learn
                        auto inputConnectionData = inputConnection->getTrainingData();
                        
                        const size_t inputWeightVar =
                        context->allocateOrReuseVariable(inputConnectionData->weight,
                        {inputConnection->getUuid(), Keys::Core::Weight});
                        
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
                        
                        const size_t outputWeightVar =
                        context->allocateOrReuseVariable(outputConnectionData->weight,
                        {outputConnection->getUuid(), Keys::Core::Weight});
                        
                        const size_t outputResponsibilityVar =
                        context->allocateOrReuseVariable(outputNeuronData->errorResponsibility,
                        {outputNeuron->getUuid(), Keys::Core::ErrorResponsibility});
                        
                        if (outputConnection->getGateNeuron() != nullptr)
                        {
                            const size_t outputGainVar =
                            context->allocateOrReuseVariable(outputConnectionData->gain,
                            {outputConnection->getUuid(), Keys::Core::Gain});
                            
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
                        
                        const size_t eligibilityVar =
                        context->allocateOrReuseVariable(target->eligibility[inputConnection->getUuid()],
                        {target->getUuid(), inputConnection->getUuid(), Keys::Core::Eligibility});
                        
                        const size_t inputWeightVar =
                        context->allocateOrReuseVariable(inputConnectionData->weight,
                        {inputConnection->getUuid(), Keys::Core::Weight});
                        
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
                        const std::string gatedNeuronId = i.first;
                        const Neuron::Ptr gatedNeuron = target->neighbours[gatedNeuronId];
                        const auto gatedNeuronData = gatedNeuron->getTrainingData();
                        
                        const size_t influenceTempVar =
                        context->allocateOrReuseVariable(0.0,
                        {Keys::Core::Influence});
                        
                        const size_t gatedNeuronOldStateVar =
                        context->allocateOrReuseVariable(gatedNeuronData->oldState,
                        {gatedNeuron->getUuid(), Keys::Core::OldState});
                        
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
                            
                            const size_t inputActivationVar =
                            context->allocateOrReuseVariable(inputNeuronData->activation,
                            {inputNeuron->getUuid(), Keys::Core::Activation});
                            
                            const size_t inputWeightVar =
                            context->allocateOrReuseVariable(inputConnectionData->weight,
                            {inputConnection->getUuid(), Keys::Core::Weight});
                            
                            hardcoded->trainProgram << influenceTempVar << " += " << inputWeightVar << " * " << inputActivationVar << std::endl;
                        }
                        
                        const size_t gatedResponsibilityVar =
                        context->allocateOrReuseVariable(gatedNeuronData->errorResponsibility,
                        {gatedNeuron->getUuid(), Keys::Core::ErrorResponsibility});
                        
                        // eq. 22
                        hardcoded->trainProgram << responsibilityVar << " += " << gatedResponsibilityVar << " * " << influenceTempVar << std::endl;
                    }
                    
                    hardcoded->trainProgram << responsibilityVar << " *= " << derivativeVar << std::endl;
                    
                    // adjust all the neuron's incoming connections
                    for (auto &i : target->incomingConnections)
                    {
                        const std::string inputConnectionUuid = i.first;
                        const Neuron::Connection::Ptr inputConnection = i.second;
                        
                        const size_t gradientTempVar =
                        context->allocateOrReuseVariable(0.0,
                        {Keys::Core::Gradient});
                        
                        hardcoded->trainProgram << gradientTempVar << " = 0" << std::endl;
                        
                        for (auto &ext : target->extended)
                        {
                            // extended elegibility trace
                            const std::string neighbourNeuronId = ext.first;
                            Neuron::EligibilityMap &xtrace = ext.second;
                            Neuron::Ptr neighbour = target->neighbours[neighbourNeuronId];
                            const auto neighbourData = neighbour->getTrainingData();
                            
                            const size_t neighbourResponsibilityVar =
                            context->allocateOrReuseVariable(neighbourData->errorResponsibility,
                            {neighbourNeuronId, Keys::Core::ErrorResponsibility});
                            
                            const size_t extendedTraceVar =
                            context->allocateOrReuseVariable(xtrace[inputConnection->getUuid()],
                            {target->getUuid(), neighbourNeuronId, inputConnectionUuid, Keys::Core::ExtendedTrace});
                            
                            hardcoded->trainProgram << gradientTempVar << " += " << neighbourResponsibilityVar << " * " << extendedTraceVar << std::endl;
                        }
                        
                        // adjust weights - aka learn
                        auto inputConnectionData = inputConnection->getTrainingData();
                        
                        const size_t inputWeightVar =
                        context->allocateOrReuseVariable(inputConnectionData->weight,
                        {inputConnection->getUuid(), Keys::Core::Weight});
                        
                        hardcoded->trainProgram << inputWeightVar << " += " << rateVar << " * " << gradientTempVar << std::endl;
                    }
                }
            }
            
            // adjust bias
            const size_t biasVar =
            context->allocateOrReuseVariable(targetData->bias,
            {target->getUuid(), Keys::Core::Bias});
            
            hardcoded->trainProgram << biasVar << " += " << rateVar << " * " << responsibilityVar << std::endl;
        }
        
        return hardcoded;
    }
    
    inline void HardcodedNeuron::restoreNeuronState(Neuron::Ptr target, HardcodedTrainingContext::Ptr context)
    {
        auto targetData = target->getTrainingData();

        const double bias = context->evaluateVariable({target->getUuid(), Keys::Core::Bias}, targetData->bias);
        const double state = context->evaluateVariable({target->getUuid(), Keys::Core::State}, targetData->state);
        const double oldState = context->evaluateVariable({target->getUuid(), Keys::Core::OldState}, targetData->oldState);
        const double activation = context->evaluateVariable({target->getUuid(), Keys::Core::Activation}, targetData->activation);
        
        targetData->bias = bias;
        targetData->state = state;
        targetData->oldState = oldState;
        targetData->activation = activation;
        
        for (auto &i : target->eligibility)
        {
            const std::string &inputConnectionUuid = i.first;
            target->eligibility[inputConnectionUuid] =
            context->evaluateVariable({target->getUuid(), inputConnectionUuid, Keys::Core::Eligibility},
                                      target->eligibility[inputConnectionUuid]);
        }
        
        for (auto &i : target->extended)
        {
            const std::string &neighbourNeuronUuid = i.first;
            Neuron::EligibilityMap &map = i.second;
            
            for (auto &j : map)
            {
                const std::string &inputConnectionUuid = i.first;
                
                const double extendedTrace =
                context->evaluateVariable({target->getUuid(), neighbourNeuronUuid, inputConnectionUuid, Keys::Core::ExtendedTrace},
                                          target->extended[neighbourNeuronUuid][inputConnectionUuid]);
                
                target->extended[neighbourNeuronUuid][inputConnectionUuid] = extendedTrace;
            }
        }
        
        for (auto &i : target->outgoingConnections)
        {
            auto outgoingConnection = i.second;
            auto outgoingConnectionUuid = i.first;
            auto outgoingConnectionData = outgoingConnection->getTrainingData();
            
            outgoingConnectionData->weight = context->evaluateVariable({outgoingConnectionUuid, Keys::Core::Weight},
                                                                       outgoingConnectionData->weight);
            
            outgoingConnectionData->gain = context->evaluateVariable({outgoingConnectionUuid, Keys::Core::Gain},
                                                                     outgoingConnectionData->gain);
        }
        
        if (target->isSelfConnected())
        {
            auto selfConnection = target->getSelfConnection();
            auto selfConnectionData = selfConnection->getTrainingData();
            
            selfConnectionData->weight = context->evaluateVariable({selfConnection->getUuid(), Keys::Core::Weight},
                                                                   selfConnectionData->weight);
            
            selfConnectionData->gain = context->evaluateVariable({selfConnection->getUuid(), Keys::Core::Gain},
                                                                 selfConnectionData->gain);
        }
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
}

#endif // TINYRNN_OPENCL_ACCELERATION

#endif // HARDCODEDNEURON_H_INCLUDED
