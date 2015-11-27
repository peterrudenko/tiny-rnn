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

#ifndef TINYRNN_SERIALIZATIONKEYS_H_INCLUDED
#define TINYRNN_SERIALIZATIONKEYS_H_INCLUDED

#include "Uuid.h"

namespace TinyRNN
{
    namespace Keys
    {
        namespace Core
        {
            static const std::string Network = "Network";
            static const std::string Layer = "Layer";
            static const std::string Layers = "Layers";
            static const std::string InputLayer = "InputLayer";
            static const std::string OutputLayer = "OutputLayer";
            static const std::string HiddenLayers = "HiddenLayers";
            static const std::string Neuron = "Neuron";
            static const std::string Neurons = "Neurons";
            static const std::string Connection = "Connection";
            static const std::string Connections = "Connections";
            static const std::string InputNeuronUuid = "InputNeuronUuid";
            static const std::string OutputNeuronUuid = "OutputNeuronUuid";
            static const std::string GateNeuronUuid = "GateNeuronUuid";
            static const std::string Uuid = "Uuid";
            static const std::string Name = "Name";
            
            static const std::string TrainingContext = "TrainingContext";
            static const std::string TrainingNeuronContext = "TrainingNeuronContext";
            static const std::string NeuronContexts = "NeuronContexts";
            static const std::string TrainingConnectionContext = "TrainingConnectionContext";
            static const std::string ConnectionContexts = "ConnectionContexts";
            static const std::string NeuronUuid = "NeuronUuid";
            static const std::string ConnectionUuid = "ConnectionUuid";
            
            static const std::string Weight = "Weight";
            static const std::string Gain = "Gain";
            
            static const std::string Bias = "Bias";
            static const std::string Activation = "Activation";
            static const std::string Derivative = "Derivative";
            static const std::string State = "State";
            static const std::string OldState = "OldState";
            
            static const std::string ErrorResponsibility = "ErrorResponsibility";
            static const std::string ProjectedActivity = "ProjectedActivity";
            static const std::string GatingActivity = "GatingActivity";
            
            static const std::string Rate = "Rate";
            static const std::string Influence = "Influence";
            static const std::string Eligibility = "Eligibility";
            static const std::string ExtendedTrace = "ExtendedTrace";
            static const std::string Target = "Target";
            
            static const std::string ErrorAccumulator = "ErrorAccumulator";
            static const std::string Gradient = "Gradient";
        }
        
        namespace Mapping
        {
            static const Uuid::Type Network = 1;
            static const Uuid::Type Layer = 2;
            static const Uuid::Type Layers = 3;
            static const Uuid::Type InputLayer = 4;
            static const Uuid::Type OutputLayer = 5;
            static const Uuid::Type HiddenLayers = 6;
            static const Uuid::Type Neuron = 7;
            static const Uuid::Type Neurons = 8;
            static const Uuid::Type Connection = 9;
            static const Uuid::Type Connections = 10;
            static const Uuid::Type InputNeuronUuid = 11;
            static const Uuid::Type OutputNeuronUuid = 12;
            static const Uuid::Type GateNeuronUuid = 13;
            static const Uuid::Type Uuid = 14;
            static const Uuid::Type Name = 15;
            
            static const Uuid::Type TrainingContext = 16;
            static const Uuid::Type TrainingNeuronContext = 17;
            static const Uuid::Type NeuronContexts = 18;
            static const Uuid::Type TrainingConnectionContext = 19;
            static const Uuid::Type ConnectionContexts = 20;
            static const Uuid::Type NeuronUuid = 21;
            static const Uuid::Type ConnectionUuid = 22;
            
            static const Uuid::Type Weight = 23;
            static const Uuid::Type Gain = 24;
            
            static const Uuid::Type Bias = 25;
            static const Uuid::Type Activation = 26;
            static const Uuid::Type Derivative = 27;
            static const Uuid::Type State = 28;
            static const Uuid::Type OldState = 29;
            
            static const Uuid::Type ErrorResponsibility = 30;
            static const Uuid::Type ProjectedActivity = 31;
            static const Uuid::Type GatingActivity = 32;
            
            static const Uuid::Type Rate = 33;
            static const Uuid::Type Influence = 34;
            static const Uuid::Type Eligibility = 35;
            static const Uuid::Type ExtendedTrace = 36;
            static const Uuid::Type Target = 37;
            
            static const Uuid::Type ErrorAccumulator = 38;
            static const Uuid::Type Gradient = 39;
        }
        
        namespace Hardcoded
        {
            static const std::string Network = "HardcodedNetwork";
            static const std::string Layer = "HardcodedLayer";
            static const std::string Layers = "HardcodedLayers";
            static const std::string Neuron = "HardcodedNeuron";
            
            static const std::string TrainingContext = "HardcodedTrainingContext";
            
            static const std::string FeedKernels = "FeedKernels";
            static const std::string FeedKernel = "FeedKernel";
            static const std::string TrainKernels = "TrainKernels";
            static const std::string TrainKernel = "TrainKernel";
            
            static const std::string NumExpressions = "NumExpressions";
            static const std::string FullSource = "FullSource";
            static const std::string EntryPoint = "EntryPoint";
            
            static const std::string IsBuilt = "IsBuilt";
            static const std::string KernelBinaries = "KernelBinaries";
            static const std::string KernelBinary = "KernelBinary";
            static const std::string KernelSentence = "KernelSentence";
            static const std::string KernelLine = "KernelLine";
            static const std::string Content = "Content";
            
            static const std::string FeedChunk = "FeedChunk";
            static const std::string TrainChunk = "TrainChunk";
            static const std::string TraceChunk = "TraceChunk";

            static const std::string RawMemory = "RawMemory";
            static const std::string VariablesMapping = "VariablesMapping";
            static const std::string MemorySize = "MemorySize";
            static const std::string OutputsSize = "OutputsSize";
            static const std::string Variable = "Variable";
            static const std::string Key = "Key";
            static const std::string Index = "Index";
        }
    }
}

#endif  // TINYRNN_SERIALIZATIONKEYS_H_INCLUDED
