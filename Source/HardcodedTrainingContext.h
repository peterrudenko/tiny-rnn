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

#ifndef TINYRNN_HARDCODEDTRAININGCONTEXT_H_INCLUDED
#define TINYRNN_HARDCODEDTRAININGCONTEXT_H_INCLUDED

#include "Common.h"
#include "SerializedObject.h"
#include "Id.h"
#include <iostream>
#include <sstream>

namespace TinyRNN
{
    class KernelSentence final
    {
    public:
        
        KernelSentence() {}
        
        friend KernelSentence &operator << (KernelSentence &i, size_t index);
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
    
    class HardcodedTrainingContext final : public SerializedObject // todo #3
    {
    public:
        
        using Ptr = std::shared_ptr<HardcodedTrainingContext>;
        using RawData = std::vector<Value>;
        using Indices = std::vector<size_t>;
        using Mapping = std::map<std::string, size_t>;
        using VariableKey = std::vector<Id>;
        
    public:
        
        HardcodedTrainingContext();
        
        Value evaluateVariable(const VariableKey &variableKey, Value defaultValue);
        size_t allocateOrReuseVariable(Value value, const VariableKey &variableKey);
        
        void registerInputVariable(size_t variableIndex);
        void registerOutputVariable(size_t variableIndex);
        void registerTargetVariable(size_t variableIndex);
        void registerRateVariable(size_t variableIndex);
        
        std::string buildInputsExpressions() const;
        std::string buildOutputsExpressions() const;
        std::string buildTargetsExpressions() const;
        std::string buildRateExpression() const;
        
        RawData &getMemory();
        RawData &getOutputs();
        
        void clear();
        
    public:
        
        virtual void deserialize(SerializationContext::Ptr context) override;
        virtual void serialize(SerializationContext::Ptr context) const override;
        
    private:
        
        RawData memory;                         // the actual data passed to the kernel
        Mapping mapping;                        // variable name connected to its index in memory
        
    private: // temporary stuff, never serialized:
        
        RawData outputs;                        // holds the most recent output
        Indices inputVariables;                 // indexes of input variables
        Indices outputVariables;                // indexes of output variables
        Indices targetVariables;                // indexes of target variables
        size_t rateVariable;
        
    private:
        
        std::string getKeyForVariable(const VariableKey &variableKey) const;
        
        friend class HardcodedNeuron;
        
        TINYRNN_DISALLOW_COPY_AND_ASSIGN(HardcodedTrainingContext);
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
    
    inline KernelSentence &operator << (KernelSentence &i, size_t index)
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
    // HardcodedTrainingContext implementation
    //===------------------------------------------------------------------===//
    
    inline HardcodedTrainingContext::HardcodedTrainingContext() : rateVariable(0)
    {}
    
    inline size_t HardcodedTrainingContext::allocateOrReuseVariable(Value value, const VariableKey &variableKey)
    {
        const std::string &key = this->getKeyForVariable(variableKey);
        const bool variableExists = (this->mapping.find(key) != this->mapping.end());
        
        if (variableExists)
        {
            const size_t variableIndex = this->mapping[key];
            this->memory[variableIndex] = value;
            return variableIndex;
        }
        else
        {
            //std::cout << this->memory.size() << " is " << key << std::endl;
            this->memory.push_back(value);
            const size_t variableIndex = (this->memory.size() - 1);
            this->mapping[key] = variableIndex;
            return variableIndex;
        }
        
        return 0;
    }
    
    inline Value HardcodedTrainingContext::evaluateVariable(const VariableKey &variableKey, Value defaultValue)
    {
        const std::string &key = this->getKeyForVariable(variableKey);
        const bool variableExists = (this->mapping.find(key) != this->mapping.end());
        
        if (variableExists)
        {
            const size_t variableIndex = this->mapping[key];
            //std::cout << "Variable: " << key << " = " << std::to_string(this->memory[variableIndex]) << std::endl;
            return this->memory[variableIndex];
        }
        
        //std::cout << "Variable missing: " << key << ", default to " << std::to_string(defaultValue) << std::endl;
        return defaultValue;
    }
    
    inline std::string HardcodedTrainingContext::getKeyForVariable(const VariableKey &variableKey) const
    {
        std::ostringstream key;
        
        std::copy(variableKey.begin(),
                  variableKey.end() - 1,
                  std::ostream_iterator<Id>(key, "::"));
        
        key << variableKey.back();
        
        return key.str();
    }
    
    inline void HardcodedTrainingContext::registerInputVariable(size_t variableIndex)
    {
        this->inputVariables.push_back(variableIndex);
    }
    
    inline void HardcodedTrainingContext::registerOutputVariable(size_t variableIndex)
    {
        this->outputVariables.push_back(variableIndex);
        this->outputs.resize(this->outputVariables.size());
    }
    
    inline void HardcodedTrainingContext::registerTargetVariable(size_t variableIndex)
    {
        this->targetVariables.push_back(variableIndex);
    }
    
    inline void HardcodedTrainingContext::registerRateVariable(size_t variableIndex)
    {
        this->rateVariable = variableIndex;
    }
    
    inline std::string HardcodedTrainingContext::buildInputsExpressions() const
    {
        KernelSentence sentence;
        
        for (size_t i = 0; i < this->inputVariables.size(); ++i)
        {
            sentence << this->inputVariables[i] << " = input[" << std::to_string(i) << "]"<< std::endl;
        }
        
        return sentence.build();
    }
    
    inline std::string HardcodedTrainingContext::buildOutputsExpressions() const
    {
        KernelSentence sentence;
        
        for (size_t i = 0; i < this->outputVariables.size(); ++i)
        {
            sentence << "output[" << std::to_string(i) << "] = " << this->outputVariables[i] << std::endl;
        }
        
        return sentence.build();
    }
    
    inline std::string HardcodedTrainingContext::buildTargetsExpressions() const
    {
        KernelSentence sentence;
        
        for (size_t i = 0; i < this->targetVariables.size(); ++i)
        {
            sentence << this->targetVariables[i] << " = target[" << std::to_string(i) << "]"<< std::endl;
        }
        
        return sentence.build();
    }
    
    inline std::string HardcodedTrainingContext::buildRateExpression() const
    {
        KernelSentence sentence;
        sentence << rateVariable << " = rate[0]" << std::endl;
        return sentence.build();
    }
    
    inline HardcodedTrainingContext::RawData &HardcodedTrainingContext::getMemory()
    {
        return this->memory;
    }
    
    inline HardcodedTrainingContext::RawData &HardcodedTrainingContext::getOutputs()
    {
        return this->outputs;
    }
    
    inline void HardcodedTrainingContext::clear()
    {
        this->memory.clear();
        this->outputs.clear();
        this->mapping.clear();
        this->inputVariables.clear();
        this->outputVariables.clear();
        this->targetVariables.clear();
        this->rateVariable = 0;
    }
    
    //===------------------------------------------------------------------===//
    // Serialization
    //===------------------------------------------------------------------===//
    
    inline void HardcodedTrainingContext::deserialize(SerializationContext::Ptr context)
    {
        this->clear();
        
        const std::string &memoryEncoded = context->getStringProperty(Keys::Hardcoded::RawMemory);
        const size_t memorySize = context->getNumberProperty(Keys::Hardcoded::MemorySize);
        
        this->memory.resize(memorySize);
        const std::vector<unsigned char> &memoryDecoded = context->decodeBase64(memoryEncoded);
        memcpy(this->memory.data(), memoryDecoded.data(), sizeof(Value) * memorySize);
        
        const size_t outputsSize = context->getNumberProperty(Keys::Hardcoded::OutputsSize);
        this->outputs.resize(outputsSize);
        
        SerializationContext::Ptr mappingNode(context->getChildContext(Keys::Hardcoded::VariablesMapping));
        
        for (size_t i = 0; i < mappingNode->getNumChildrenContexts(); ++i)
        {
            SerializationContext::Ptr variableNode(mappingNode->getChildContext(i));
            const std::string &key = variableNode->getStringProperty(Keys::Hardcoded::Key);
            const size_t index = variableNode->getNumberProperty(Keys::Hardcoded::Index);
            this->mapping[key] = index;
        }
    }
    
    inline void HardcodedTrainingContext::serialize(SerializationContext::Ptr context) const
    {
        const std::string memoryEncoded = context->encodeBase64((const unsigned char *)this->memory.data(), sizeof(Value) * this->memory.size());
        context->setStringProperty(memoryEncoded, Keys::Hardcoded::RawMemory);
        context->setNumberProperty(this->memory.size(), Keys::Hardcoded::MemorySize);
        context->setNumberProperty(this->outputs.size(), Keys::Hardcoded::OutputsSize);
        
        SerializationContext::Ptr mappingNode(context->addChildContext(Keys::Hardcoded::VariablesMapping));
        
        for (const auto &i : this->mapping)
        {
            SerializationContext::Ptr variableNode(mappingNode->addChildContextUnordered(Keys::Hardcoded::Variable));
            variableNode->setStringProperty(i.first, Keys::Hardcoded::Key);
            variableNode->setNumberProperty(i.second, Keys::Hardcoded::Index);
        }
    }
}

#endif  // TINYRNN_HARDCODEDTRAININGCONTEXT_H_INCLUDED
