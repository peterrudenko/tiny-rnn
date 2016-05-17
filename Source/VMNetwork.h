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
#include "VMNeuron.h"
#include "HardcodedTrainingContext.h"
#include "Id.h"
#include "ScopedMemoryBlock.h"
#include "ScopedTimer.h"
#include "SerializedObject.h"

namespace TinyRNN
{
    class VMNetwork final : public SerializedObject
    {
    public:
        
        using Ptr = std::shared_ptr<VMNetwork>;
        using VMLayers = std::vector<VMNeuron::Vector>;
        
    public:
        
        explicit VMNetwork(HardcodedTrainingContext::Ptr targetContext);
        VMNetwork(HardcodedTrainingContext::Ptr targetContext, VMLayers targetLayers);
        
        bool compile();
        
        HardcodedTrainingContext::Ptr getContext() const noexcept;
        
        HardcodedTrainingContext::RawData feed(const HardcodedTrainingContext::RawData &values);
        void train(Value rate, const HardcodedTrainingContext::RawData &target);
        
    public:
        
        virtual void deserialize(SerializationContext::Ptr context) override;
        virtual void serialize(SerializationContext::Ptr context) const override;
        
    private:
        
        HardcodedTrainingContext::Ptr trainingContext;
        
    private:
        
#if TINYRNN_OPENCL_ACCELERATION
        
        cl::Device clDevice;
        cl::Context clContext;
        cl::Program clProgram;
        cl::CommandQueue clQueue;
        
        cl::Buffer clMemoryBuffer;
        cl::Buffer clInputsBuffer;
        cl::Buffer clOutputsBuffer;
        cl::Buffer clTargetsBuffer;
        cl::Buffer clRateBuffer;
        
#endif
      
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
            
#if TINYRNN_OPENCL_ACCELERATION
            cl::Kernel clKernel;
            cl::Buffer clCommandsBuffer;
            cl::Buffer clIndicesBuffer;
#endif
            
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
        
        TINYRNN_DISALLOW_COPY_AND_ASSIGN(VMNetwork);
    };
    
    //===------------------------------------------------------------------===//
    // HardcodedNetwork implementation
    //===------------------------------------------------------------------===//
    
    inline VMNetwork::VMNetwork(HardcodedTrainingContext::Ptr targetContext) :
    trainingContext(targetContext)
    {
        VMLayers empty;
        this->initialize(empty);
    }
    
    inline VMNetwork::VMNetwork(HardcodedTrainingContext::Ptr targetContext,
                                VMLayers targetLayers) :
    trainingContext(targetContext)
    {
        this->initialize(targetLayers);
    }
    
    inline HardcodedTrainingContext::Ptr VMNetwork::getContext() const noexcept
    {
        return this->trainingContext;
    }
    
    //===------------------------------------------------------------------===//
    // Compiling
    //===------------------------------------------------------------------===//
    
    inline bool VMNetwork::initialize(const VMLayers &targetLayers)
    {
        const ScopedTimer timer("VMNetwork::initialize");
        
#if TINYRNN_OPENCL_ACCELERATION
        
        std::vector<cl::Platform> allPlatforms;
        cl::Platform::get(&allPlatforms);
        
        if (allPlatforms.size() == 0)
        {
            std::cout << "No OpenCL platforms found!\n";
            return false;
        }
        
        const cl::Platform defaultPlatform = allPlatforms.front();
        std::cout << "OpenCL platform: " << defaultPlatform.getInfo<CL_PLATFORM_NAME>() << "\n";
        
        std::vector<cl::Device> allDevices;
        defaultPlatform.getDevices(CL_DEVICE_TYPE_ALL, &allDevices);
        
        if (allDevices.empty())
        {
            std::cout << "No OpenCL devices found!\n";
            return false;
        }
        
        this->clDevice = allDevices.front();
        std::cout << "Using OpenCL device: " << this->clDevice.getInfo<CL_DEVICE_NAME>() << "\n";
        
        this->clContext = cl::Context(this->clDevice);
        
#endif
        
        this->feedKernel = this->compileFeedKernel(targetLayers);
        this->trainKernel = this->compileTrainKernel(targetLayers);
        
        return true;
    }
    
    inline bool VMNetwork::compile()
    {
        const ScopedTimer timer("VMNetwork::compile");
        
#if TINYRNN_OPENCL_ACCELERATION
        
        cl::Program::Sources clSources;
        clSources.push_back({this->feedKernel->fullSource.c_str(), this->feedKernel->fullSource.length()});
        clSources.push_back({this->trainKernel->fullSource.c_str(), this->trainKernel->fullSource.length()});
        
        this->clProgram = cl::Program(this->clContext, clSources);
        
        if (this->clProgram.build({this->clDevice}) != CL_SUCCESS)
        {
            std::cout << " Error building: " << this->clProgram.getBuildInfo<CL_PROGRAM_BUILD_LOG>(this->clDevice) << std::endl;
            return false;
        }
        else
        {
            std::cout << "Build ok, variables count: " << this->trainingContext->getMemory().size() << std::endl;
        }
        
        {
            const ScopedTimer feedTimer("Compiling feed kernel");
            this->feedKernel->clKernel = cl::Kernel(this->clProgram, this->feedKernel->entryPoint.c_str());
            this->feedKernel->clCommandsBuffer =
            cl::Buffer(this->clContext,
                       CL_MEM_READ_ONLY | CL_MEM_USE_HOST_PTR,
                       sizeof(char) * this->feedKernel->commands.size(),
                       (void *)this->feedKernel->commands.data());
            this->feedKernel->clIndicesBuffer =
            cl::Buffer(this->clContext,
                       CL_MEM_READ_ONLY | CL_MEM_USE_HOST_PTR,
                       sizeof(Index) * this->feedKernel->indices.size(),
                       (void *)this->feedKernel->indices.data());
            this->feedKernel->isBuilt = true;
        }
        
        {
            const ScopedTimer feedTimer("Compiling train kernel");
            this->trainKernel->clKernel = cl::Kernel(this->clProgram, this->trainKernel->entryPoint.c_str());
            this->trainKernel->clCommandsBuffer =
            cl::Buffer(this->clContext,
                       CL_MEM_READ_ONLY | CL_MEM_USE_HOST_PTR,
                       sizeof(char) * this->trainKernel->commands.size(),
                       (void *)this->trainKernel->commands.data());
            this->trainKernel->clIndicesBuffer =
            cl::Buffer(this->clContext,
                       CL_MEM_READ_ONLY | CL_MEM_USE_HOST_PTR,
                       sizeof(Index) * this->trainKernel->indices.size(),
                       (void *)this->trainKernel->indices.data());
            this->trainKernel->isBuilt = true;
        }
        
        this->clQueue = cl::CommandQueue(this->clContext, this->clDevice);
        
        this->clMemoryBuffer =
        cl::Buffer(this->clContext,
                   CL_MEM_READ_WRITE | CL_MEM_USE_HOST_PTR,
                   sizeof(Value) * this->trainingContext->getMemory().size(),
                   (void *)this->trainingContext->getMemory().data());
        
        return true;
        
#else
        
        return false;
        
#endif
    }
    
#define VALUE_STRING std::string((sizeof(Value) == sizeof(double)) ? "double" : "float")
    
    inline bool VMNetwork::isBuilt() const
    {
        return (!this->feedKernel->isBuilt && !this->trainKernel->isBuilt);
    }
    
    //===------------------------------------------------------------------===//
    // Compiling all the expressions
    //===------------------------------------------------------------------===//
    
    void vmProcess(const char *commands, uint32_t numCommands,
                   const Index *indices, Value *registers)
    {
        uint32_t c = 0; // command number
        uint32_t i = 0; // index number
        
#define X(INDEX) (registers[indices[i + INDEX]])
        
        while (c < numCommands)
        {
            switch (commands[c])
            {
                case VMProgram::Zero:
                    X(0) = 0;
                    i += 1;
                    break;
                case VMProgram::Activation:
                    X(0) = (1.0 / (1.0 + exp(-X(1))));
                    i += 2;
                    break;
                case VMProgram::Derivative:
                    X(0) = X(1) * (1.0 - X(1));
                    i += 2;
                    break;
                case VMProgram::AAP:
                    X(0) += X(1) * X(2);
                    i += 3;
                    break;
                case VMProgram::AAPP:
                    X(0) += X(1) * X(2) * X(3);
                    i += 4;
                    break;
                case VMProgram::A:
                    X(0) = X(1);
                    i += 2;
                    break;
                case VMProgram::AS:
                    X(0) = X(1) + X(2);
                    i += 3;
                    break;
                case VMProgram::AD:
                    X(0) = X(1) - X(2);
                    i += 3;
                    break;
                case VMProgram::AP:
                    X(0) = X(1) * X(2);
                    i += 3;
                    break;
                case VMProgram::APP:
                    X(0) = X(1) * X(2) * X(3);
                    i += 4;
                    break;
                case VMProgram::APS:
                    X(0) = X(1) * X(2) + X(3);
                    i += 4;
                    break;
                case VMProgram::APSP:
                    X(0) = X(1) * X(2) + X(3) * X(4);
                    i += 5;
                    break;
                case VMProgram::APPS:
                    X(0) = X(1) * X(2) * X(3) + X(4);
                    i += 5;
                    break;
                case VMProgram::APPSP:
                    X(0) = X(1) * X(2) * X(3) + X(4) * X(5);
                    i += 6;
                    break;
                case VMProgram::APPSPP:
                    X(0) = X(1) * X(2) * X(3) + X(4) * X(5) * X(6);
                    i += 7;
                    break;
                default:
                    break;
            }
            c++;
        }
    }
    
    static std::string kVMProcessingKernel =
    "\
    uint32_t c = 0; // command number\
    uint32_t i = 0; // index number\
    \
    #define X(INDEX) (registers[indices[i + INDEX]])\
    \
    while (c < numCommands)\
    {\
        switch (commands[c])\
        {\
            case 0:\
                X(0) = 0;\
                i += 1;\
                break;\
            case 1:\
                X(0) = (1.0 / (1.0 + exp(-X(1))));\
                i += 2;\
                break;\
            case 2:\
                X(0) = X(1) * (1.0 - X(1));\
                i += 2;\
                break;\
            case 3:\
                X(0) += X(1) * X(2);\
                i += 3;\
                break;\
            case 4:\
                X(0) += X(1) * X(2) * X(3);\
                i += 4;\
                break;\
            case 5:\
                X(0) = X(1);\
                i += 2;\
                break;\
            case 6:\
                X(0) = X(1) + X(2);\
                i += 3;\
                break;\
            case 7:\
                X(0) = X(1) - X(2);\
                i += 3;\
                break;\
            case 8:\
                X(0) = X(1) * X(2);\
                i += 3;\
                break;\
            case 9:\
                X(0) = X(1) * X(2) * X(3);\
                i += 4;\
                break;\
            case 10:\
                X(0) = X(1) * X(2) + X(3);\
                i += 4;\
                break;\
            case 11:\
                X(0) = X(1) * X(2) + X(3) * X(4);\
                i += 5;\
                break;\
            case 12:\
                X(0) = X(1) * X(2) * X(3) + X(4);\
                i += 5;\
                break;\
            case 13:\
                X(0) = X(1) * X(2) * X(3) + X(4) * X(5);\
                i += 6;\
                break;\
            case 14:\
                X(0) = X(1) * X(2) * X(3) + X(4) * X(5) * X(6);\
                i += 7;\
                break;\
            default:\
                break;\
        }\
        c++;\
    }\
    ";
    
    inline VMNetwork::Kernel::Ptr
    VMNetwork::compileFeedKernel(const VMLayers &targetLayers) const
    {
        Kernel::Ptr kernel(new Kernel());
        kernel->entryPoint = ("feed");
        kernel->fullSource =
        "void kernel " + kernel->entryPoint +
        "(global const " + VALUE_STRING + " *input, " +
        "global " + VALUE_STRING + " *output, " +
        "global const char *commands, " +
        "global const uint numCommands, " +
        "global const uint *indices, " +
        "global " + VALUE_STRING + " *registers) {\n";
        
        kernel->fullSource += this->buildInputsExpressions();
        kernel->fullSource += kVMProcessingKernel;
        kernel->fullSource += this->buildOutputsExpressions();
        kernel->fullSource += "}\n";
        
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
        
        return kernel;
    }
    
    inline VMNetwork::Kernel::Ptr
    VMNetwork::compileTrainKernel(const VMLayers &targetLayers) const
    {
        Kernel::Ptr kernel(new Kernel());
        kernel->entryPoint = ("train");
        kernel->fullSource =
        "void kernel " + kernel->entryPoint +
        "(global const " + VALUE_STRING + " *rate, " +
        "global const " + VALUE_STRING + " *target, " +
        "global const char *commands, " +
        "global const uint numCommands, " +
        "global const uint *indices, " +
        "global " + VALUE_STRING + " *registers) {\n";
        
        kernel->fullSource += this->buildRateExpression();
        kernel->fullSource += this->buildTargetsExpressions();
        kernel->fullSource += kVMProcessingKernel;
        kernel->fullSource += "}\n";

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
        
        return kernel;
    }
    
    inline std::string VMNetwork::buildInputsExpressions() const
    {
        KernelSentence sentence;
        const auto &inputVariables = this->trainingContext->getInputVariables();
        
        for (size_t i = 0; i < inputVariables.size(); ++i)
        {
            sentence << inputVariables[i] << " = input[" << std::to_string(i) << "]"<< std::endl;
        }
        
        return sentence.build();
    }
    
    inline std::string VMNetwork::buildOutputsExpressions() const
    {
        KernelSentence sentence;
        const auto &outputVariables = this->trainingContext->getOutputVariables();
        
        for (size_t i = 0; i < outputVariables.size(); ++i)
        {
            sentence << "output[" << std::to_string(i) << "] = " << outputVariables[i] << std::endl;
        }
        
        return sentence.build();
    }
    
    inline std::string VMNetwork::buildTargetsExpressions() const
    {
        KernelSentence sentence;
        const auto &targetVariables = this->trainingContext->getTargetVariables();
        
        for (size_t i = 0; i < targetVariables.size(); ++i)
        {
            sentence << targetVariables[i] << " = target[" << std::to_string(i) << "]"<< std::endl;
        }
        
        return sentence.build();
    }
    
    inline std::string VMNetwork::buildRateExpression() const
    {
        KernelSentence sentence;
        sentence << this->trainingContext->getRateVariable() << " = rate[0]" << std::endl;
        return sentence.build();
    }
    
    //===------------------------------------------------------------------===//
    // Core
    //===------------------------------------------------------------------===//
    
    inline HardcodedTrainingContext::RawData VMNetwork::feed(const HardcodedTrainingContext::RawData &inputs)
    {
        std::fill(this->trainingContext->getOutputs().begin(),
                  this->trainingContext->getOutputs().end(),
                  0.0);
        
#if TINYRNN_OPENCL_ACCELERATION
        
        this->clInputsBuffer = cl::Buffer(this->clContext,
                                          CL_MEM_READ_ONLY | CL_MEM_USE_HOST_PTR,
                                          sizeof(Value) * inputs.size(),
                                          (void *)inputs.data());
        
        this->clOutputsBuffer = cl::Buffer(this->clContext,
                                           CL_MEM_WRITE_ONLY | CL_MEM_USE_HOST_PTR,
                                           sizeof(Value) * this->trainingContext->getOutputs().size(),
                                           (void *)this->trainingContext->getOutputs().data());
        
        this->feedKernel->clKernel.setArg(0, this->clInputsBuffer);
        this->feedKernel->clKernel.setArg(1, this->clOutputsBuffer);
        this->feedKernel->clKernel.setArg(2, this->feedKernel->clCommandsBuffer);
        this->feedKernel->clKernel.setArg(3, this->feedKernel->clIndicesBuffer);
        this->feedKernel->clKernel.setArg(4, this->clMemoryBuffer);
        this->clQueue.enqueueNDRangeKernel(this->feedKernel->clKernel, cl::NullRange, cl::NDRange(1), cl::NullRange);
        this->clQueue.finish();
        
        
#else
        
        // TODO handle inputs and outputs
        
        vmProcess(this->feedKernel->commands.data(),
                  this->feedKernel->commands.size(),
                  this->feedKernel->indices.data(),
                  this->trainingContext->getMemory().data());
        
#endif
        
        return this->trainingContext->getOutputs();
    }
    
    inline void VMNetwork::train(Value rate, const HardcodedTrainingContext::RawData &targets)
    {
#if TINYRNN_OPENCL_ACCELERATION
        
        this->clTargetsBuffer = cl::Buffer(this->clContext,
                                           CL_MEM_READ_ONLY | CL_MEM_USE_HOST_PTR,
                                           sizeof(Value) * targets.size(),
                                           (void *)targets.data());
        
        this->clRateBuffer = cl::Buffer(this->clContext,
                                        CL_MEM_READ_ONLY | CL_MEM_USE_HOST_PTR,
                                        sizeof(Value),
                                        (void *)&rate);
        
        this->trainKernel->clKernel.setArg(0, this->clRateBuffer);
        this->trainKernel->clKernel.setArg(1, this->clTargetsBuffer);
        this->trainKernel->clKernel.setArg(2, this->trainKernel->clCommandsBuffer);
        this->trainKernel->clKernel.setArg(3, this->trainKernel->clIndicesBuffer);
        this->trainKernel->clKernel.setArg(4, this->clMemoryBuffer);
        this->clQueue.enqueueNDRangeKernel(this->trainKernel->clKernel, cl::NullRange, cl::NDRange(1), cl::NullRange);
        this->clQueue.finish();
        
#else
        
        // TODO handle rate and targets
        
        vmProcess(this->feedKernel->commands.data(),
                  this->feedKernel->commands.size(),
                  this->feedKernel->indices.data(),
                  this->trainingContext->getMemory().data());
        
#endif
    }
    
    //===------------------------------------------------------------------===//
    // Serialization
    //===------------------------------------------------------------------===//
    
    inline void VMNetwork::deserialize(SerializationContext::Ptr context)
    {
        this->feedKernel = nullptr;
        this->trainKernel = nullptr;
        // TODO
    }
    
    inline void VMNetwork::serialize(SerializationContext::Ptr context) const
    {
        // TODO
    }
    
    inline void VMNetwork::Kernel::deserialize(SerializationContext::Ptr context)
    {
        // TODO
    }
    
    inline void VMNetwork::Kernel::serialize(SerializationContext::Ptr context) const
    {
        // TODO
    }
} // namespace TinyRNN

#endif // TINYRNN_VMNETWORK_H_INCLUDED
