#pragma once

#include <cctype>
#include <cmath>
#include <stack>
#include <stdexcept>
#include <string>
#include <format>
#include <type_traits>
#include <variant>
#include <array>
#include <vector>
#include <unordered_map>


namespace Matematyka {

namespace rpn {

enum class UnitType {
    Number,
    Variable,
    Function,
    Operator,
};

enum class OperationType {
    Add,
    Substract,
    Multiply,
    Divide,
    Power,
    FunctionApplication,
};

class Operation {
public:
    Operation() = delete;

    static Operation create(const char op) {
        switch (op) {
            case '+': return Operation(OperationType::Add);
            case '-': return Operation(OperationType::Substract);
            case '*': return Operation(OperationType::Multiply);
            case '/': return Operation(OperationType::Divide);
            case '^': return Operation(OperationType::Power);
            case '#': return Operation(OperationType::FunctionApplication);
        }
        throw std::runtime_error("Unsupported operation");
    }

    static bool IsOperator(const char op) {
        return
            op == '*' || op == '/' || op == '%' ||
            op == '+' || op == '-' ||
            op == '^' || op == '#' ||
            op == '(' || op == ')';
    }

    static int Priority(char c) {
        switch (c) {
            case '#': return 5;
            case '^': return 4;
            case '*': case '/': case '%': return 3;
            case '+': case '-': return 2;
            case '(': return 1;
            default: return 0;
        }
    }

    char toChar() const {
        switch (type) {
            case OperationType::Add: return '+';
            case OperationType::Substract: return '-';
            case OperationType::Multiply: return '*';
            case OperationType::Divide: return '/';
            case OperationType::Power: return '^';
            case OperationType::FunctionApplication: return '#';
        }
        throw std::runtime_error("Broken operation");
    }

    OperationType getType() const { return type; }

    bool operator==(const Operation& other) const = default;

private:
    OperationType type;
    Operation(OperationType type) : type(type) {}
};

using TVarId = std::string;

#define MATH_FN(NAME) \
    FunctionHolder<T>{ MakeWrapper<T>([](T x) { return std::NAME(x); }), #NAME }

template<class T>
using TFunction = T (*)(T);

template<class T>
struct FunctionHolder {
    TFunction<T> function;
    std::string_view name;

    bool operator==(const FunctionHolder& other) const = default;
};

template<class T, class F>
constexpr TFunction<T> MakeWrapper(F function) {
    if constexpr (std::is_convertible_v<F, TFunction<T>>) {
        return function;
    } else {
        static_assert(std::is_empty_v<F> && std::is_default_constructible_v<F>,
                      "Callable with state cannot be converted to a raw function pointer");
        return [](T value) -> T { return static_cast<T>(F{}(value)); };
    }
}

template<class T>
constexpr auto FUNCTIONS_AVAILABLE = std::to_array<FunctionHolder<T>>({
    MATH_FN(sin), MATH_FN(cos), MATH_FN(tan),
    MATH_FN(exp), MATH_FN(log),
    MATH_FN(floor), MATH_FN(ceil),
});

template<class T>
constexpr const FunctionHolder<T>* FindFunction(std::string_view name) {
    for (const auto& f : FUNCTIONS_AVAILABLE<T>)
        if (f.name == name) return &f;
    return nullptr;
}

static_assert(FindFunction<double>("cos") != nullptr);
static_assert(FindFunction<double>("nope") == nullptr);


template<class T>
class Unit {
public:
    using TValue = std::variant<
        TVarId,
        T,
        FunctionHolder<T>,
        Operation
    >;

    Unit() = delete;
    Unit(const Unit& other) = default;
    Unit& operator=(const Unit& other) = default;
    Unit(Unit&& other) = default;

    static Unit createScalar(const T& x) {
        return Unit(UnitType::Number, x);
    }

    static Unit createVariable(const TVarId id) {
        return Unit(UnitType::Variable, id);
    }

    static Unit createFunction(const std::string_view& s) {
        const auto* function = FindFunction<T>(s);
        if (!function) {
            throw std::runtime_error(std::string("Unsupported function ") + std::string(s));
        }
        return Unit(UnitType::Function, *function);
    }

    static Unit createOperator(const char op) {
        return Unit(UnitType::Operator, Operation::create(op));
    }

    template<class V>
    requires (
        std::is_same_v<V, char>
        || std::is_convertible_v<V, std::string>
        || std::is_convertible_v<V, T>
    )
    static Unit create(const V& x) {
        if constexpr (std::is_convertible_v<V, std::string>) {
            if (FindFunction<T>(x) != nullptr) {
                return createFunction(x);
            }
            return createVariable(x);
        } else if constexpr (std::is_same_v<V, char>) {
            return createOperator(x);
        } else {
            return createScalar(x);
        }
    }

    std::string toString() const {
        switch (type) {
            case UnitType::Number: {
                if constexpr (std::is_floating_point_v<T>) {
                    std::string s = std::format("{:.4f}", std::get<T>(value));
                    s.erase(s.find_last_not_of('0') + 1, std::string::npos);
                    if (!s.empty() && s.back() == '.') s.pop_back();
                    return s;
                } else {
                    return std::to_string(std::get<T>(value));
                }
            }
            case UnitType::Function: return std::string(std::get<FunctionHolder<T>>(value).name);
            case UnitType::Operator: return std::string(1, std::get<Operation>(value).toChar());
            case UnitType::Variable: return std::get<TVarId>(value);
            default:
                throw std::runtime_error("Unsupported toString()");
        }
    }

    const TValue& getValue() const {
        return value;
    }

    bool operator==(const Unit& other) const = default;

    UnitType getType() const { return type; }

    const TVarId& getVariable() const { return std::get<TVarId>(value); }
    const T& getScalar() const { return std::get<T>(value); }
    const FunctionHolder<T>& getFunction() const { return std::get<FunctionHolder<T>>(value); }
    const Operation& getOperation() const { return std::get<Operation>(value); }

private:
    UnitType type;
    TValue value;

    Unit(UnitType type, TValue value) : type(type), value(value) {}
};

}

template<class T>
class VariableHolder {
public:

    T getVariableSafe(const std::string& variable) const {
        if (auto var = variables.find(variable); var != variables.end()) {
            return var->second;
        }
        return T{};
    }

    bool hasVariable(const std::string& variable) const {
        return variables.contains(variable);
    }

    T getVariable(const std::string& variable) const {
        return variables.find(variable)->second;
    }

    void setVariable(const std::string& variable, const T& value) {
        variables[variable] = value;
    }

private:
    std::unordered_map<std::string, T> variables;
};

template<class T>
T ScalarizeUnit(const rpn::Unit<T>& unit, const VariableHolder<T>& variables) {
    switch (unit.getType()) {
        case rpn::UnitType::Variable: return variables.getVariableSafe(std::get<rpn::TVarId>(unit.getValue()));
        case rpn::UnitType::Number: return std::get<T>(unit.getValue());
        default:
            throw std::runtime_error("Can't scalarize");
    }
}

template<class T>
T ApplyPlainOperator(const rpn::Operation& op, const rpn::Unit<T>& leftUnit, const rpn::Unit<T>& rightUnit, const VariableHolder<T>& vars) {
    const auto left = ScalarizeUnit(leftUnit, vars);
    const auto right = ScalarizeUnit(rightUnit, vars);
    switch (op.getType()) {
        case rpn::OperationType::Add: return left + right;
        case rpn::OperationType::Substract: return left - right;
        case rpn::OperationType::Multiply: return left * right;
        case rpn::OperationType::Divide: return left / right;
        case rpn::OperationType::Power: return std::pow(left, right);
        default:
            throw std::runtime_error("Can't apply");
    }
}

template<class T>
T ApplyFunction(const rpn::Unit<T> functionUnit, const rpn::Unit<T>& argumentUnit, const VariableHolder<T>& vars) {
    const auto argument = ScalarizeUnit(argumentUnit, vars);
    const auto function = std::get<rpn::FunctionHolder<T>>(functionUnit.getValue());
    return function.function(argument);
}

/// MAIN

template<class T>
using RpnHolder = std::vector<rpn::Unit<T>>;

template<class T>
struct RpnHash {
    std::size_t operator()(const RpnHolder<T>& rpn) const {
        std::size_t seed = rpn.size();
        const auto mix = [&seed](const std::size_t value) {
            seed ^= value + 0x9e3779b97f4a7c15ULL + (seed << 6) + (seed >> 2);
        };
        for (const auto& unit : rpn) {
            mix(static_cast<std::size_t>(unit.getType()));
            switch (unit.getType()) {
                case rpn::UnitType::Number:
                    mix(std::hash<T>{}(unit.getScalar()));
                    break;
                case rpn::UnitType::Variable:
                    mix(std::hash<std::string>{}(unit.getVariable()));
                    break;
                case rpn::UnitType::Function:
                    mix(std::hash<const void*>{}(
                        reinterpret_cast<const void*>(unit.getFunction().function)));
                    break;
                case rpn::UnitType::Operator:
                    mix(static_cast<std::size_t>(unit.getOperation().getType()));
                    break;
            }
        }
        return seed;
    }
};

class ExpressionValidator {
public:
    enum class Error {
        NoError,
        ParenthesesMismatch,
        ConsecutiveOperators,
    };

    struct Result {
        Error error = Error::NoError;
        std::size_t position = 0;

        std::string getMessage() const {
            switch (error) {
                case Error::NoError: return "Ok";
                case Error::ParenthesesMismatch: return "Parentheses mismatch at position " + std::to_string(position);
                case Error::ConsecutiveOperators: return "Consecutive operators at position " + std::to_string(position);
                default:
                    return "unknown";
            }
        }
    };

    static Result Validate(const std::string& expression) {
        int parCount = 0;
        bool lastWasOperator = true;

        for (std::size_t i = 0; i < expression.size(); ++i) {
            const auto c = expression[i];
            if (std::isspace(c)) {
                continue;
            }


            if (c == '(') {
                ++parCount;
                lastWasOperator = true;
            } else if (c == ')') {
                if (--parCount < 0) {
                    return Result{
                        .error = Error::ParenthesesMismatch,
                        .position = i,
                    };
                }
                lastWasOperator = false;
            } else if (rpn::Operation::IsOperator(c)) {
                if (lastWasOperator && c != '-' && c != '(' && c != '+') {
                    return Result{
                        .error = Error::ConsecutiveOperators,
                        .position = i,
                    };
                }
                lastWasOperator = true;
            } else {
                lastWasOperator = false;
            }
        }
        if (parCount != 0) {
            return Result {
                .error = Error::ParenthesesMismatch,
                .position = expression.size(),
            };
        }
        return Result{
            .error = Error::NoError,
        };
    }
};

template<class T>
class Expression {
public:
    using TRpn = RpnHolder<T>;
    using TUnit = rpn::Unit<T>;

    Expression() = default;
    Expression(const Expression& other) = default;
    Expression(Expression&& other) noexcept : expression(std::move(other.expression)) {}
    Expression& operator=(Expression&& other) = default;
    explicit Expression(const TRpn& rpn) : expression(rpn) {}
    explicit Expression(TRpn&& rpn) : expression(std::move(rpn)) {}
    explicit Expression(const std::string& expr) { setExpression(expr); }

    static std::string SimplifyExpresssion(const std::string& original) {
        std::string result;
        result.reserve(original.size());

        for (std::size_t i = 0; i < original.size(); ++i) {
            if (std::isspace(original[i])) {
                continue;
            }

            // Unary plus and minus
            if ((original[i] == '-' || original[i] == '+')
                && (i == 0 || original[i - 1] == '(' || rpn::Operation::IsOperator(original[i - 1]))) {
                result += '0';
            }

            // Function and variable names
            if (std::isalpha(original[i])) {
                std::size_t size = 0;
                while (i + size < original.size() && std::isalnum(original[i + size])) {
                    ++size;
                }
                const auto name = original.substr(i, size);
                const auto isFunction = (rpn::FindFunction<T>(name) != nullptr);
                result += name;
                if (isFunction) {
                    result += '#';
                }
                i += size - 1; // because is going to be incremented
            } else {
                result += original[i];
            }
        }
        return result;
    }


    static TRpn MakeRpn(const std::string& s) {
        std::stack<char> stack;
        TRpn result;
        std::string buffer;

        const auto flushBuffer = [&result](std::string& buffer) -> void {
            if (buffer.empty()) {
                return;
            }

            if (std::isdigit(buffer.front()) || buffer.front() == '.') {
                try {
                    const T value = static_cast<T>(std::stod(buffer));
                    result.push_back(
                        TUnit::createScalar(value)
                    );
                } catch (...) {
                    throw std::runtime_error("Invalid number " + buffer);
                }
            } else {
                const auto isFunction = (rpn::FindFunction<T>(buffer) != nullptr);
                if (isFunction) {
                    result.push_back(
                        TUnit::createFunction(buffer)
                    );
                } else {
                    result.push_back(
                        TUnit::createVariable(buffer)
                    );
                }
            }
            buffer.clear();
        };

        for (const char c : s) {
            if (c == '(') {
                flushBuffer(buffer);
                stack.push(c);
            } else if (c == ')') {
                flushBuffer(buffer);
                while (!stack.empty() && stack.top() != '(') {
                    result.push_back(TUnit::create(stack.top()));
                    stack.pop();
                }
                if (stack.empty()) {
                    throw std::runtime_error("Mismatched parenthesis");
                }
                stack.pop(); // remove '('
            } else if (rpn::Operation::IsOperator(c)) {
                flushBuffer(buffer);
                while (!stack.empty()
                    && rpn::Operation::Priority(stack.top()) >= rpn::Operation::Priority(c)) {
                    result.push_back(TUnit::create(stack.top()));
                    stack.pop();
                }
                stack.push(c);
            } else {
                buffer += c;
            }
        }

        flushBuffer(buffer);
        while (!stack.empty()) {
            if (stack.top() == '(') {
                throw std::runtime_error("Parentheses mismatch");
            }
            result.push_back(TUnit::create(stack.top()));
            stack.pop();
        }
        return result;
    }

    T run(const VariableHolder<T>& variables) const {
        using UnitType = rpn::UnitType;
        using Operation = rpn::Operation;
        using OperationType = rpn::OperationType;

        if (expression.empty()) {
            throw std::runtime_error("No expression defined");
        }

        std::stack<TUnit> stack;
        for (const TUnit& unit : expression) {
            if (unit.getType() != UnitType::Operator) {
                stack.push(
                    unit.getType() == UnitType::Variable
                    ? TUnit::create(ScalarizeUnit(unit, variables))
                    : unit
                );
                continue;
            }

            if (stack.size() < 2) {
                throw std::runtime_error("Ill-formed RPN");
            }

            const auto right = stack.top(); stack.pop();
            const auto left  = stack.top(); stack.pop();
            const auto operation = std::get<Operation>(unit.getValue());
            const auto result =
                operation.getType() == OperationType::FunctionApplication
                ? ApplyFunction(left, right, variables)
                : ApplyPlainOperator(operation, left, right, variables);

            stack.push(TUnit::create(result));
        }

        if (stack.size() != 1) {
            throw std::runtime_error("Ill formed RPN: stack imbalance");
        }
        const auto unitType = stack.top().getType();
        if (unitType != UnitType::Number) {
            throw std::runtime_error("Last value is not a number");
        }
        return std::get<T>(stack.top().getValue());
    }

    void setExpression(const std::string& original) {
        const auto validationResult = ExpressionValidator::Validate(original);
        if (validationResult.error != ExpressionValidator::Error::NoError) {
            throw std::runtime_error(validationResult.getMessage());
        }

        expression = MakeRpn(SimplifyExpresssion(original));
    }

    const TRpn& getRpn() const { return expression; }
    TRpn& getRpnMutable() { return expression; }

    std::string toString() const {
        static const auto precedence = [](const rpn::OperationType op) {
            switch (op) {
                case rpn::OperationType::Add:
                case rpn::OperationType::Substract:
                    return 1;
                case rpn::OperationType::Multiply:
                case rpn::OperationType::Divide:
                    return 2;
                case rpn::OperationType::Power:
                    return 3;
                default:
                    return 100;
            }
        };

        struct Node {
            std::string text;
            int prec;
        };

        static const auto wrap = [](const Node& child, int parentPrec, bool needStrict) {
            const bool need = child.prec < parentPrec || (needStrict && child.prec == parentPrec);
            return need ? "(" + child.text + ")" : child.text;
        };


        const auto rawSequence = [this] {
            std::string out = "<ill-formed:";
            for (const TUnit& unit : expression) {
                out += ' ';
                out += unit.toString();
            }
            out += '>';
            return out;
        };

        std::stack<Node> stack;

        for (const TUnit& unit : expression) {
            if (unit.getType() == rpn::UnitType::Operator) {
                const auto op = unit.getOperation();
                if (stack.size() < 2) {
                    return rawSequence();
                }

                if (op.getType() == rpn::OperationType::FunctionApplication) {
                    const Node right = stack.top(); stack.pop();
                    const Node left  = stack.top(); stack.pop();
                    stack.push(Node{ left.text + "(" + right.text + ")", 100 });
                    continue;
                }

                const Node right = stack.top(); stack.pop();
                const Node left  = stack.top(); stack.pop();
                const int  p     = precedence(op.getType());

                const bool rightStrict = (op.getType() == rpn::OperationType::Substract || op.getType() == rpn::OperationType::Divide);
                const std::string l = wrap(left,  p, false);
                const std::string r = wrap(right, p, rightStrict);
                stack.push(Node{ l + unit.toString() + r, p });
                continue;
            }
            int leafPrec = 100;
            if (unit.getType() == rpn::UnitType::Number && unit.getScalar() < 0) {
                leafPrec = 0;
            }
            stack.push(Node{ unit.toString(), leafPrec });
        }
        if (stack.size() != 1) {
            return rawSequence();
        }
        return stack.top().text;
    }

private:
    TRpn expression;

};

}
