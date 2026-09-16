#include <doctest/doctest.h>

#include "core/lib/rpn.h"
#include "core/lib/rpn_compile.h"

#include <bit>
#include <cstdint>
#include <random>
#include <string>
#include <vector>

namespace {

using Expression = Matematyka::Expression<double>;
using Unit = Expression::TUnit;
using Rpn = Expression::TRpn;
using Compiled = Matematyka::CompiledExpression<double>;
using Status = Compiled::Status;

// Slot 0 is the permanent zero slot, standing in for what
// VariableHolder::getVariableSafe returns for a name it has never been told.
constexpr std::uint32_t kZeroSlot = 0;
constexpr std::uint32_t kSlotOfX = 1;

std::uint32_t SlotOf(const std::string& name) {
    return name == "x" ? kSlotOfX : kZeroSlot;
}

double CompiledRun(const Rpn& rpn, const double x) {
    Compiled program;
    REQUIRE(program.compile(rpn, SlotOf) == Status::Ok);
    const double values[] = {0.0, x};
    std::vector<double> stack(program.getStackDepth() + 1);
    return program.eval(values, stack.data());
}

// A random well-formed RPN, built the way growTree builds one: operands first,
// then the operator that consumes them.
void Grow(Rpn& out, std::mt19937& rng, const int depthLeft) {
    std::uniform_real_distribution<double> unit(0.0, 1.0);
    if (depthLeft == 0 || unit(rng) < 0.35) {
        if (unit(rng) < 0.5) {
            out.push_back(Unit::createScalar(
                std::uniform_real_distribution<double>{-5.0, 5.0}(rng)));
        } else {
            out.push_back(Unit::createVariable(unit(rng) < 0.8 ? "x" : "unknown"));
        }
        return;
    }
    if (unit(rng) < 0.3) {
        static constexpr const char* kNames[] = {"sin", "cos", "tan", "exp", "log", "floor", "ceil"};
        out.push_back(Unit::createFunction(kNames[std::uniform_int_distribution<int>{0, 6}(rng)]));
        Grow(out, rng, depthLeft - 1);
        out.push_back(Unit::createOperator('#'));
        return;
    }
    static constexpr char kOperators[] = {'+', '-', '*', '/', '^'};
    Grow(out, rng, depthLeft - 1);
    Grow(out, rng, depthLeft - 1);
    out.push_back(Unit::createOperator(kOperators[std::uniform_int_distribution<int>{0, 4}(rng)]));
}

}  // namespace


TEST_CASE("A compiled expression computes what the RPN says") {
    Rpn rpn;
    rpn.push_back(Unit::createVariable("x"));
    rpn.push_back(Unit::createScalar(2));
    rpn.push_back(Unit::createOperator('*'));
    CHECK(CompiledRun(rpn, 3) == doctest::Approx(6));

    Rpn call;
    call.push_back(Unit::createFunction("sin"));
    call.push_back(Unit::createVariable("x"));
    call.push_back(Unit::createOperator('#'));
    CHECK(CompiledRun(call, 0.5) == doctest::Approx(std::sin(0.5)));
}

TEST_CASE("An unknown variable reads from the zero slot") {
    Rpn rpn;
    rpn.push_back(Unit::createVariable("nowhere"));
    rpn.push_back(Unit::createScalar(1));
    rpn.push_back(Unit::createOperator('+'));
    CHECK(CompiledRun(rpn, 99) == doctest::Approx(1));
}

TEST_CASE("Compilation rejects exactly what the interpreter throws on") {
    const auto statusOf = [](const Rpn& rpn) {
        Compiled program;
        return program.compile(rpn, SlotOf);
    };

    CHECK(statusOf(Rpn{}) == Status::Empty);

    Rpn missing;
    missing.push_back(Unit::createScalar(1));
    missing.push_back(Unit::createOperator('+'));
    CHECK(statusOf(missing) == Status::MissingOperand);

    Rpn leftover;
    leftover.push_back(Unit::createScalar(1));
    leftover.push_back(Unit::createScalar(2));
    CHECK(statusOf(leftover) == Status::LeftoverOperands);

    Rpn loneFunction;
    loneFunction.push_back(Unit::createFunction("sin"));
    CHECK(statusOf(loneFunction) == Status::ResultNotANumber);

    Rpn functionUnderOperator;
    functionUnderOperator.push_back(Unit::createFunction("sin"));
    functionUnderOperator.push_back(Unit::createScalar(2));
    functionUnderOperator.push_back(Unit::createOperator('+'));
    CHECK(statusOf(functionUnderOperator) == Status::OperandNotScalar);

    Rpn appliedNumber;
    appliedNumber.push_back(Unit::createScalar(2));
    appliedNumber.push_back(Unit::createScalar(3));
    appliedNumber.push_back(Unit::createOperator('#'));
    CHECK(statusOf(appliedNumber) == Status::AppliedNonFunction);
}

TEST_CASE("The value stack only has to hold values, not function units") {
    Rpn call;
    call.push_back(Unit::createFunction("sin"));
    call.push_back(Unit::createVariable("x"));
    call.push_back(Unit::createOperator('#'));

    Compiled program;
    REQUIRE(program.compile(call, SlotOf) == Status::Ok);
    CHECK(program.getStackDepth() == 1);
}

TEST_CASE("The compiled evaluator is bit-identical to the interpreter") {
    // The whole justification for a second evaluator: it must agree with the
    // reference on every expression and every input, down to the bit pattern --
    // bit_cast rather than ==, so a signed zero or a NaN payload cannot hide a
    // divergence. Both well-formed and deliberately damaged RPNs are compared,
    // because agreeing about what *fails* matters just as much: the ranker turns
    // every interpreter throw into a rank of exactly zero.
    std::mt19937 rng(20260916);
    Matematyka::VariableHolder<double> holder;
    Compiled program;
    std::vector<double> stack;

    std::size_t compared = 0;
    std::size_t rejected = 0;
    for (int sample = 0; sample < 4000; ++sample) {
        Rpn rpn;
        Grow(rpn, rng, std::uniform_int_distribution<int>{0, 4}(rng));

        // Damage a quarter of them the way the mutations do.
        if (sample % 4 == 0 && !rpn.empty()) {
            const auto at = std::uniform_int_distribution<std::size_t>{0, rpn.size()}(rng);
            rpn.insert(rpn.begin() + at, Unit::createFunction("cos"));
        }

        for (const double x : {0.0, 1.0, -2.5, 1e8, -1e-9}) {
            holder.setVariable("x", x);
            holder.setVariable("unknown", 0.0);

            bool threw = false;
            double interpreted = 0;
            try {
                interpreted = Expression{Rpn(rpn)}.run(holder);
            } catch (...) {
                threw = true;
            }

            const auto status = program.compile(rpn, SlotOf);
            REQUIRE(threw == (status != Status::Ok));
            if (threw) {
                ++rejected;
                continue;
            }

            stack.assign(program.getStackDepth() + 1, 0.0);
            const double values[] = {0.0, x};
            const double compiled = program.eval(values, stack.data());
            CHECK(std::bit_cast<std::uint64_t>(compiled)
                  == std::bit_cast<std::uint64_t>(interpreted));
            ++compared;
        }
    }

    // Guard the guard: a corpus that never reached either branch would pass
    // vacuously.
    CHECK(compared > 5000);
    CHECK(rejected > 500);
}
