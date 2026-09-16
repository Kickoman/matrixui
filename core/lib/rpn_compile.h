#pragma once

#include "core/lib/rpn.h"

#include <cstdint>
#include <vector>

namespace Matematyka {


enum class CompiledOp : std::uint8_t {
    PushConst,
    PushVar,
    Add,
    Sub,
    Mul,
    Div,
    Pow,
    CallFunction,
};

struct CompiledInstruction {
    CompiledOp op;
    std::uint32_t argument;
};

template<class T>
class CompiledExpression {
public:
    enum class Status : std::uint8_t {
        Ok,
        Empty,
        MissingOperand,
        LeftoverOperands,
        ResultNotANumber,
        OperandNotScalar,
        AppliedNonFunction,
    };


    template<class Resolver>
    Status compile(const RpnHolder<T>& rpn, Resolver&& slotOf) {
        code.clear();
        constants.clear();
        functions.clear();
        kinds.clear();
        stackDepth = 0;

        if (rpn.empty()) {
            return status = Status::Empty;
        }

        std::uint32_t depth = 0;
        for (const rpn::Unit<T>& unit : rpn) {
            if (unit.getType() != rpn::UnitType::Operator) {
                pushLeaf(unit, slotOf, depth);
                continue;
            }

            if (kinds.size() < 2) {
                return status = Status::MissingOperand;
            }
            const auto right = kinds.back(); kinds.pop_back();
            const auto left = kinds.back(); kinds.pop_back();
            const auto operation = unit.getOperation().getType();

            if (operation == rpn::OperationType::FunctionApplication) {
                if (right.isFunction) {
                    return status = Status::OperandNotScalar;
                }
                if (!left.isFunction) {
                    return status = Status::AppliedNonFunction;
                }
                code.push_back(CompiledInstruction{CompiledOp::CallFunction, left.function});
                kinds.push_back(Kind{});
                continue;
            }

            if (left.isFunction || right.isFunction) {
                return status = Status::OperandNotScalar;
            }
            code.push_back(CompiledInstruction{OpOf(operation), 0});
            --depth;
            kinds.push_back(Kind{});
        }

        if (kinds.size() != 1) {
            return status = Status::LeftoverOperands;
        }
        if (kinds.front().isFunction) {
            return status = Status::ResultNotANumber;
        }
        return status = Status::Ok;
    }

    Status getStatus() const noexcept { return status; }
    std::uint32_t getStackDepth() const noexcept { return stackDepth; }

    T eval(const T* values, T* stack) const noexcept {
        T* top = stack;
        for (const CompiledInstruction& instruction : code) {
            switch (instruction.op) {
                case CompiledOp::PushConst:
                    *top++ = constants[instruction.argument];
                    break;
                case CompiledOp::PushVar:
                    *top++ = values[instruction.argument];
                    break;
                case CompiledOp::Add:
                    top[-2] = top[-2] + top[-1]; --top;
                    break;
                case CompiledOp::Sub:
                    top[-2] = top[-2] - top[-1]; --top;
                    break;
                case CompiledOp::Mul:
                    top[-2] = top[-2] * top[-1]; --top;
                    break;
                case CompiledOp::Div:
                    top[-2] = top[-2] / top[-1]; --top;
                    break;
                case CompiledOp::Pow:
                    top[-2] = std::pow(top[-2], top[-1]); --top;
                    break;
                case CompiledOp::CallFunction:
                    top[-1] = functions[instruction.argument](top[-1]);
                    break;
            }
        }
        return stack[0];
    }

private:
    struct Kind {
        bool isFunction = false;
        std::uint32_t function = 0;
    };

    static CompiledOp OpOf(const rpn::OperationType operation) {
        switch (operation) {
            case rpn::OperationType::Add: return CompiledOp::Add;
            case rpn::OperationType::Substract: return CompiledOp::Sub;
            case rpn::OperationType::Multiply: return CompiledOp::Mul;
            case rpn::OperationType::Divide: return CompiledOp::Div;
            case rpn::OperationType::Power: return CompiledOp::Pow;
            case rpn::OperationType::FunctionApplication: break;
        }
        return CompiledOp::Add;
    }

    template<class Resolver>
    void pushLeaf(const rpn::Unit<T>& unit, Resolver&& slotOf, std::uint32_t& depth) {
        switch (unit.getType()) {
            case rpn::UnitType::Number:
                code.push_back(CompiledInstruction{
                    CompiledOp::PushConst, static_cast<std::uint32_t>(constants.size())});
                constants.push_back(unit.getScalar());
                break;
            case rpn::UnitType::Variable:
                code.push_back(CompiledInstruction{CompiledOp::PushVar, slotOf(unit.getVariable())});
                break;
            case rpn::UnitType::Function:
                functions.push_back(unit.getFunction().function);
                kinds.push_back(Kind{
                    .isFunction = true,
                    .function = static_cast<std::uint32_t>(functions.size() - 1),
                });
                return;
            case rpn::UnitType::Operator:
                return;
        }
        kinds.push_back(Kind{});
        ++depth;
        stackDepth = std::max(stackDepth, depth);
    }

    std::vector<CompiledInstruction> code;
    std::vector<T> constants;
    std::vector<rpn::TFunction<T>> functions;
    std::vector<Kind> kinds;
    Status status = Status::Empty;
    std::uint32_t stackDepth = 0;
};

}  // namespace Matematyka
