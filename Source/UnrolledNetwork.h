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

#ifndef TINYRNN_VMNETWORK_H_INCLUDED
#define TINYRNN_VMNETWORK_H_INCLUDED

#include "Common.h"
#include "UnrolledNeuron.h"
#include "UnrolledTrainingContext.h"
#include "Id.h"
#include "ScopedMemoryBlock.h"
#include "ScopedTimer.h"
#include "SerializedObject.h"

namespace TinyRNN
{
    class UnrolledNetwork final : public SerializedObject
    {
    public:
        
        using Ptr = std::shared_ptr<UnrolledNetwork>;
        using VMLayers = std::vector<UnrolledNeuron::Vector>;
        
    public:
        
        explicit UnrolledNetwork(UnrolledTrainingContext::Ptr targetContext);
        UnrolledNetwork(UnrolledTrainingContext::Ptr targetContext, VMLayers targetLayers);
        
        UnrolledTrainingContext::Ptr getContext() const noexcept;
        
        UnrolledTrainingContext::RawData feed(const UnrolledTrainingContext::RawData &values);
        void train(Value rate, const UnrolledTrainingContext::RawData &target);
        
    public:
        
        virtual void deserialize(SerializationContext::Ptr context) override;
        virtual void serialize(SerializationContext::Ptr context) const override;
        
    private:
        
        UnrolledTrainingContext::Ptr trainingContext;
        
    private:
      
        class Kernel final : public SerializedObject
        {
        public:
            
            using Ptr = std::shared_ptr<Kernel>;
            
            Kernel() : isBuilt(false) {}
            
            bool isBuilt;
            std::string fullSource;
            std::string entryPoint;
            
            std::vector<char> commands;
            std::vector<Index> indices; // Index is the same type as cl_uint
            
        public:
            
            virtual void deserialize(SerializationContext::Ptr context) override;
            virtual void serialize(SerializationContext::Ptr context) const override;
            
        private:
            
            TINYRNN_DISALLOW_COPY_AND_ASSIGN(Kernel);
        };
        
        Kernel::Ptr feedKernel;
        Kernel::Ptr trainKernel;
        
        Kernel::Ptr compileFeedKernel(const VMLayers &targetLayers) const;
        Kernel::Ptr compileTrainKernel(const VMLayers &targetLayers) const;
        
        std::string buildInputsExpressions() const;
        std::string buildOutputsExpressions() const;
        std::string buildTargetsExpressions() const;
        std::string buildRateExpression() const;
        
        bool initialize(const VMLayers &targetLayers);
        bool isBuilt() const;
        
        TINYRNN_DISALLOW_COPY_AND_ASSIGN(UnrolledNetwork);
    };
    
    //===------------------------------------------------------------------===//
    // HardcodedNetwork implementation
    //===------------------------------------------------------------------===//
    
    inline UnrolledNetwork::UnrolledNetwork(UnrolledTrainingContext::Ptr targetContext) :
    trainingContext(targetContext)
    {
        VMLayers empty;
        this->initialize(empty);
    }
    
    inline UnrolledNetwork::UnrolledNetwork(UnrolledTrainingContext::Ptr targetContext,
                                VMLayers targetLayers) :
    trainingContext(targetContext)
    {
        this->initialize(targetLayers);
    }
    
    inline UnrolledTrainingContext::Ptr UnrolledNetwork::getContext() const noexcept
    {
        return this->trainingContext;
    }
    
    //===------------------------------------------------------------------===//
    // Compiling
    //===------------------------------------------------------------------===//
    
    inline bool UnrolledNetwork::initialize(const VMLayers &targetLayers)
    {
        const ScopedTimer timer("UnrolledNetwork::initialize");
        
        this->feedKernel = this->compileFeedKernel(targetLayers);
        this->trainKernel = this->compileTrainKernel(targetLayers);
        
        return true;
    }
    
#define VALUE_STRING std::string((sizeof(Value) == sizeof(double)) ? "double" : "float")
    
    inline bool UnrolledNetwork::isBuilt() const
    {
        return (!this->feedKernel->isBuilt && !this->trainKernel->isBuilt);
    }
    
    //===------------------------------------------------------------------===//
    // Compiling all the expressions
    //===------------------------------------------------------------------===//
    
    static void vmProcess(const char *commands,
                          const Index *indices,
                          Value *registers)
    {
        uint32_t c = 0; // command number
        uint32_t i = 0; // index number
        char command = 0;
        
#define I(INDEX) (indices[i + INDEX])
#define X(INDEX) (registers[indices[i + INDEX]])
#define SKIP(NUMBER) (i += NUMBER)
        
        while (command != VMProgram::End)
        {
            switch (command = commands[c++])
            {
                case VMProgram::Zero:
                    X(0) = 0;
                    SKIP(1);
                    break;
                case VMProgram::Clip:
                    X(0) = std::max(-1.f, std::min(X(0), 1.f));
                    SKIP(1);
                    break;
                case VMProgram::Activation:
                    X(0) = X(1) > 0.0 ? X(1) : (0.01 * X(1));
                    SKIP(2);
                    break;
                case VMProgram::Derivative:
                    X(0) = X(1) > 0.0 ? 1.0 : 0.01;
                    SKIP(2);
                    break;
                case VMProgram::AAP:
                    X(0) = X(0) + X(1) * X(2);
                    SKIP(3);
                    break;
                case VMProgram::AAPP:
                    X(0) = X(0) + X(1) * X(2) * X(3);
                    SKIP(4);
                    break;
                case VMProgram::A:
                    X(0) = X(1);
                    SKIP(2);
                    break;
                case VMProgram::AS:
                    X(0) = X(1) + X(2);
                    SKIP(3);
                    break;
                case VMProgram::AD:
                    X(0) = X(1) - X(2);
                    SKIP(3);
                    break;
                case VMProgram::AP:
                    X(0) = X(1) * X(2);
                    SKIP(3);
                    break;
                case VMProgram::APP:
                    X(0) = X(1) * X(2) * X(3);
                    SKIP(4);
                    break;
                case VMProgram::APS:
                    X(0) = X(1) * X(2) + X(3);
                    SKIP(4);
                    break;
                case VMProgram::APSP:
                    X(0) = X(1) * X(2) + X(3) * X(4);
                    SKIP(5);
                    break;
                case VMProgram::APPS:
                    X(0) = X(1) * X(2) * X(3) + X(4);
                    SKIP(5);
                    break;
                case VMProgram::APPSP:
                    X(0) = X(1) * X(2) * X(3) + X(4) * X(5);
                    SKIP(6);
                    break;
                case VMProgram::APPSPP:
                    X(0) = X(1) * X(2) * X(3) + X(4) * X(5) * X(6);
                    SKIP(7);
                    break;
                case VMProgram::FeedState:
                {
                    const auto loopCount = I(0);
                    const auto stateIndex = I(1);
                    SKIP(2);
                    
                    for (Index loop = 0; loop < loopCount; ++loop)
                    {
                        registers[stateIndex] = registers[stateIndex] + X(0) * X(1) * X(2);
                        SKIP(3);
                    }
                    
                    break;
                }
                    
                default:
                    break;
            }
        }
    }
    
    static std::string kVMProcessingKernel =
    "\
    uint c = 0;\
    uint i = 0;\
    char command = 0;\
    \
    while (command != 127)\
    {\
        switch (command = commands[c++])\
        {\
            case 0: {\
                x[id[i+0]] = 0;\
                i += 1;\
                break;\
            }\
            case 1: {\
                x[id[i+0]] = x[id[i+0]] < 0.0 ? 0.0 : (x[id[i+0]] > 1.0 ? 1.0 : x[id[i+0]]);\
                i += 1;\
                break;\
            }\
            case 2: {\
                x[id[i+0]] = x[id[i+1]] > 0.0 ? x[id[i+1]] : (0.01 * x[id[i+1]]);\
                i += 2;\
                break;\
            }\
            case 3: {\
                x[id[i+0]] = x[id[i+1]] > 0.0 ? 1.0 : 0.01;\
                i += 2;\
                break;\
            }\
            case 4: {\
                x[id[i+0]] += x[id[i+1]] * x[id[i+2]];\
                i += 3;\
                break;\
            }\
            case 5: {\
                x[id[i+0]] += x[id[i+1]] * x[id[i+2]] * x[id[i+3]];\
                i += 4;\
                break;\
            }\
            case 6: {\
                x[id[i+0]] = x[id[i+1]];\
                i += 2;\
                break;\
            }\
            case 7: {\
                x[id[i+0]] = x[id[i+1]] + x[id[i+2]];\
                i += 3;\
                break;\
            }\
            case 8: {\
                x[id[i+0]] = x[id[i+1]] - x[id[i+2]];\
                i += 3;\
                break;\
            }\
            case 9: {\
                x[id[i+0]] = x[id[i+1]] * x[id[i+2]];\
                i += 3;\
                break;\
            }\
            case 10: {\
                x[id[i+0]] = x[id[i+1]] * x[id[i+2]] * x[id[i+3]];\
                i += 4;\
                break;\
            }\
            case 11: {\
                x[id[i+0]] = x[id[i+1]] * x[id[i+2]] + x[id[i+3]];\
                i += 4;\
                break;\
            }\
            case 12: {\
                x[id[i+0]] = x[id[i+1]] * x[id[i+2]] + x[id[i+3]] * x[id[i+4]];\
                i += 5;\
                break;\
            }\
            case 13: {\
                x[id[i+0]] = x[id[i+1]] * x[id[i+2]] * x[id[i+3]] + x[id[i+4]];\
                i += 5;\
                break;\
            }\
            case 14: {\
                x[id[i+0]] = x[id[i+1]] * x[id[i+2]] * x[id[i+3]] + x[id[i+4]] * x[id[i+5]];\
                i += 6;\
                break;\
            }\
            case 15: {\
                x[id[i+0]] = x[id[i+1]] * x[id[i+2]] * x[id[i+3]] + x[id[i+4]] * x[id[i+5]] * x[id[i+6]];\
                i += 7;\
                break;\
            }\
            case 16:\
            {\
                const uint loopCount = id[i+0];\
                const uint stateIndex = id[i+1];\
                i += 2;\
                for (uint loop = 0; loop < loopCount; ++loop)\
                {\
                    x[id[stateIndex]] = x[id[stateIndex]] + (x[id[i+0]] * x[id[i+1]] * x[id[i+2]]);\
                    i += 3;\
                }\
                break;\
            }\
        }\
    }\
    ";
    
    inline UnrolledNetwork::Kernel::Ptr
    UnrolledNetwork::compileFeedKernel(const VMLayers &targetLayers) const
    {
        Kernel::Ptr kernel(new Kernel());
        kernel->entryPoint = "feed";
        kernel->fullSource =
        "void kernel " + kernel->entryPoint +
        "(global const " + VALUE_STRING + " *input, " +
        "global " + VALUE_STRING + " *output, " +
        "global const char *commands, " +
        "global const uint *id, " +
        "global " + VALUE_STRING + " *x) {\n";
        
        kernel->fullSource += this->buildInputsExpressions();
        kernel->fullSource += kVMProcessingKernel;
        kernel->fullSource += this->buildOutputsExpressions();
        kernel->fullSource += "}\n";
        
        //std::cout << kernel->fullSource << std::endl;
        
        for (const auto &layer : targetLayers)
        {
            for (const auto &neuron : layer)
            {
                const auto &feedCommands = neuron->getFeedChunk().commands;
                const auto &traceCommands = neuron->getTraceChunk().commands;
                kernel->commands.reserve(kernel->commands.size() + feedCommands.size() + traceCommands.size());
                kernel->commands.insert(kernel->commands.end(), feedCommands.begin(), feedCommands.end());
                kernel->commands.insert(kernel->commands.end(), traceCommands.begin(), traceCommands.end());
                
                const auto &feedIndices = neuron->getFeedChunk().indices;
                const auto &traceIndices = neuron->getTraceChunk().indices;
                kernel->indices.reserve(kernel->indices.size() + feedIndices.size() + traceIndices.size());
                kernel->indices.insert(kernel->indices.end(), feedIndices.begin(), feedIndices.end());
                kernel->indices.insert(kernel->indices.end(), traceIndices.begin(), traceIndices.end());
            }
        }
        
        kernel->commands.push_back(VMProgram::End);
        return kernel;
    }
    
    inline UnrolledNetwork::Kernel::Ptr
    UnrolledNetwork::compileTrainKernel(const VMLayers &targetLayers) const
    {
        Kernel::Ptr kernel(new Kernel());
        kernel->entryPoint = ("train");
        kernel->fullSource =
        "void kernel " + kernel->entryPoint +
        "(global const " + VALUE_STRING + " *rate, " +
        "global const " + VALUE_STRING + " *target, " +
        "global const char *commands, " +
        "global const uint *id, " +
        "global " + VALUE_STRING + " *x) {\n";
        
        kernel->fullSource += this->buildRateExpression();
        kernel->fullSource += this->buildTargetsExpressions();
        kernel->fullSource += kVMProcessingKernel;
        kernel->fullSource += "}\n";
        
        //std::cout << kernel->fullSource << std::endl;

        for (size_t l = targetLayers.size(); l --> 0 ;)
        {
            const auto &layer = targetLayers[l];
            
            for (size_t n = layer.size(); n --> 0 ;)
            {
                const auto &neuron = layer[n];
                const auto &trainCommands = neuron->getTrainChunk().commands;
                const auto &trainIndices = neuron->getTrainChunk().indices;
                kernel->commands.reserve(kernel->commands.size() + trainCommands.size());
                kernel->commands.insert(kernel->commands.end(), trainCommands.begin(), trainCommands.end());
                kernel->indices.reserve(kernel->indices.size() + trainIndices.size());
                kernel->indices.insert(kernel->indices.end(), trainIndices.begin(), trainIndices.end());
            }
        }
        
        kernel->commands.push_back(VMProgram::End);
        return kernel;
    }
    
    inline std::string UnrolledNetwork::buildInputsExpressions() const
    {
        std::stringstream sentence;
        const auto &inputVariables = this->trainingContext->getInputVariables();
        
        for (size_t i = 0; i < inputVariables.size(); ++i)
        {
            sentence << "x[" << inputVariables[i]  << "] = input[" << std::to_string(i) << "];"<< std::endl;
        }
        
        return sentence.str();
    }
    
    inline std::string UnrolledNetwork::buildOutputsExpressions() const
    {
        std::stringstream sentence;
        const auto &outputVariables = this->trainingContext->getOutputVariables();
        
        for (size_t i = 0; i < outputVariables.size(); ++i)
        {
            sentence << "output[" << std::to_string(i) << "] = x[" << outputVariables[i] << "];" << std::endl;
        }
        
        return sentence.str();
    }
    
    inline std::string UnrolledNetwork::buildTargetsExpressions() const
    {
        std::stringstream sentence;
        const auto &targetVariables = this->trainingContext->getTargetVariables();
        
        for (size_t i = 0; i < targetVariables.size(); ++i)
        {
            sentence << "x[" << targetVariables[i] << "] = target[" << std::to_string(i) << "];"<< std::endl;
        }
        
        return sentence.str();
    }
    
    inline std::string UnrolledNetwork::buildRateExpression() const
    {
        std::stringstream sentence;
        sentence << "x[" << this->trainingContext->getRateVariable() << "] = rate[0];" << std::endl;
        return sentence.str();
    }
    
    //===------------------------------------------------------------------===//
    // Core
    //===------------------------------------------------------------------===//
    
    inline UnrolledTrainingContext::RawData UnrolledNetwork::feed(const UnrolledTrainingContext::RawData &inputs)
    {
        std::fill(this->trainingContext->getOutputs().begin(),
                  this->trainingContext->getOutputs().end(),
                  0.0);
        
        const auto &inputIds = this->trainingContext->getInputVariables();
        for (size_t i = 0; i < inputIds.size(); ++i)
        {
            this->trainingContext->getMemory()[inputIds[i]] = inputs[i];
        }
        
        vmProcess(this->feedKernel->commands.data(),
                  this->feedKernel->indices.data(),
                  this->trainingContext->getMemory().data());
        
        const auto &outputIds = this->trainingContext->getOutputVariables();
        for (size_t i = 0; i < outputIds.size(); ++i)
        {
            this->trainingContext->getOutputs()[i] = this->trainingContext->getMemory()[outputIds[i]];
        }
        
        return this->trainingContext->getOutputs();
    }
    
    inline void UnrolledNetwork::train(Value rate, const UnrolledTrainingContext::RawData &targets)
    {
        const auto &targetIds = this->trainingContext->getTargetVariables();
        for (size_t i = 0; i < targetIds.size(); ++i)
        {
            this->trainingContext->getMemory()[targetIds[i]] = targets[i];
        }
        
        const auto rateId = this->trainingContext->getRateVariable();
        this->trainingContext->getMemory()[rateId] = rate;
        
        vmProcess(this->trainKernel->commands.data(),
                  this->trainKernel->indices.data(),
                  this->trainingContext->getMemory().data());
    }
    
    //===------------------------------------------------------------------===//
    // Serialization
    //===------------------------------------------------------------------===//
    
    inline void UnrolledNetwork::deserialize(SerializationContext::Ptr context)
    {
        this->feedKernel = nullptr;
        this->trainKernel = nullptr;
        
        if (auto feedKernelNode = context->getChildContext(Keys::Unrolled::FeedKernel))
        {
            this->feedKernel = Kernel::Ptr(new Kernel());
            this->feedKernel->deserialize(feedKernelNode);
        }
        
        if (auto trainKernelNode = context->getChildContext(Keys::Unrolled::TrainKernel))
        {
            this->trainKernel = Kernel::Ptr(new Kernel());
            this->trainKernel->deserialize(trainKernelNode);
        }
    }
    
    inline void UnrolledNetwork::serialize(SerializationContext::Ptr context) const
    {
        SerializationContext::Ptr feedKernelNode(context->addChildContext(Keys::Unrolled::FeedKernel));
        this->feedKernel->serialize(feedKernelNode);
        
        SerializationContext::Ptr trainKernelNode(context->addChildContext(Keys::Unrolled::TrainKernel));
        this->trainKernel->serialize(trainKernelNode);
    }
    
    inline void UnrolledNetwork::Kernel::deserialize(SerializationContext::Ptr context)
    {
        const std::string &commandsEncoded = context->getStringProperty(Keys::Unrolled::Commands);
        const size_t commandsSize = context->getNumberProperty(Keys::Unrolled::CommandsSize);
        const auto &commandsDecoded = context->decodeBase64(commandsEncoded);
        
        this->commands.resize(commandsSize);
        memcpy(this->commands.data(), commandsDecoded.data(), sizeof(char) * commandsSize);
        
        const std::string &indicesEncoded = context->getStringProperty(Keys::Unrolled::Indices);
        const size_t indicesSize = context->getNumberProperty(Keys::Unrolled::IndicesSize);
        const auto &indicesDecoded = context->decodeBase64(indicesEncoded);
        
        this->indices.resize(indicesSize);
        memcpy(this->indices.data(), indicesDecoded.data(), sizeof(Index) * indicesSize);
        
        this->entryPoint = context->getStringProperty(Keys::Unrolled::EntryPoint);
        this->fullSource = context->getStringProperty(Keys::Unrolled::FullSource);
    }
    
    inline void UnrolledNetwork::Kernel::serialize(SerializationContext::Ptr context) const
    {
        const std::string commandsEncoded =
        context->encodeBase64((const unsigned char *)this->commands.data(),
                              sizeof(char) * this->commands.size());
        
        context->setStringProperty(commandsEncoded, Keys::Unrolled::Commands);
        context->setNumberProperty(this->commands.size(), Keys::Unrolled::CommandsSize);
        
        const std::string indicesEncoded =
        context->encodeBase64((const unsigned char *)this->indices.data(),
                              sizeof(Index) * this->indices.size());
        
        context->setStringProperty(indicesEncoded, Keys::Unrolled::Indices);
        context->setNumberProperty(this->indices.size(), Keys::Unrolled::IndicesSize);
        
        context->setStringProperty(this->entryPoint, Keys::Unrolled::EntryPoint);
        context->setStringProperty(this->fullSource, Keys::Unrolled::FullSource);
    }
} // namespace TinyRNN

#endif // TINYRNN_VMNETWORK_H_INCLUDED
