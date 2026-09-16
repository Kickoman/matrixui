#include <doctest/doctest.h>

#include "core/lib/rpn.h"

#include <cmath>
#include <string>
#include <variant>

namespace {

using Expression = Matematyka::Expression<double>;
using Unit = Expression::TUnit;
using Rpn = Expression::TRpn;
using Holder = Matematyka::VariableHolder<double>;

double Eval(const std::string& text, Holder& variables) {
    return Expression(text).run(variables);
}

double EvalWithX(const std::string& text, const double x) {
    Holder variables;
    variables.setVariable("x", x);
    return Eval(text, variables);
}

}  // namespace


// The evaluator is the hot path of the genetics mode, so these cases pin what
// it computes -- and, just as importantly, how it fails -- before any rewrite.

TEST_CASE("run evaluates arithmetic with the usual precedence") {
    CHECK(EvalWithX("x+1", 2) == doctest::Approx(3));
    CHECK(EvalWithX("x*2+1", 3) == doctest::Approx(7));
    CHECK(EvalWithX("1+x*2", 3) == doctest::Approx(7));
    CHECK(EvalWithX("(1+x)*2", 3) == doctest::Approx(8));
    CHECK(EvalWithX("x-1-1", 5) == doctest::Approx(3));
    CHECK(EvalWithX("x/2", 5) == doctest::Approx(2.5));
    CHECK(EvalWithX("x^2", 3) == doctest::Approx(9));
    CHECK(EvalWithX("2^x^2", 3) == doctest::Approx(std::pow(std::pow(2., 3.), 2.)));
}

TEST_CASE("run applies every function in the table") {
    CHECK(EvalWithX("sin(x)", 0.5) == doctest::Approx(std::sin(0.5)));
    CHECK(EvalWithX("cos(x)", 0.5) == doctest::Approx(std::cos(0.5)));
    CHECK(EvalWithX("tan(x)", 0.5) == doctest::Approx(std::tan(0.5)));
    CHECK(EvalWithX("exp(x)", 0.5) == doctest::Approx(std::exp(0.5)));
    CHECK(EvalWithX("log(x)", 0.5) == doctest::Approx(std::log(0.5)));
    CHECK(EvalWithX("floor(x)", 2.7) == doctest::Approx(2));
    CHECK(EvalWithX("ceil(x)", 2.1) == doctest::Approx(3));
    CHECK(EvalWithX("1+sin(x*2)", 0.5) == doctest::Approx(1. + std::sin(1.)));
}

TEST_CASE("run produces infinities and NaN rather than throwing") {
    // The ranker distinguishes these from failures: they cost the invalid-result
    // penalty, whereas a throw costs the whole organism.
    CHECK(std::isinf(EvalWithX("x/0", 1)));
    CHECK(std::isnan(EvalWithX("x/0-x/0", 1)));
    CHECK(std::isnan(EvalWithX("log(x)", -1)));
}

TEST_CASE("An unknown variable reads as zero, it is not an error") {
    // getVariableSafe falls back to T{}. Expressions seeded by hand may well
    // mention a name no data point provides, and they must still evaluate.
    Holder variables;
    variables.setVariable("x", 4);
    CHECK(Eval("y+1", variables) == doctest::Approx(1));
    CHECK(Eval("x+y", variables) == doctest::Approx(4));
}

TEST_CASE("A VariableHolder keeps values between runs") {
    // The ranker reuses one holder across every data point and writes only the
    // variables that point carries, so an absent variable keeps the previous
    // row's value. Any faster binding scheme has to reproduce that.
    Holder variables;
    variables.setVariable("x", 2);
    CHECK(Eval("x*10", variables) == doctest::Approx(20));
    CHECK(Eval("x*10", variables) == doctest::Approx(20));

    variables.setVariable("x", 3);
    CHECK(Eval("x*10", variables) == doctest::Approx(30));
}


// --- failure modes -------------------------------------------------------
// Every one of these reaches FunctionGenetizerApplier::rankOrganism as a
// caught exception worth a rank of exactly zero.

TEST_CASE("run rejects an empty expression") {
    Holder variables;
    CHECK_THROWS_WITH_AS(Expression{}.run(variables),
                         doctest::Contains("No expression defined"), std::runtime_error);
}

TEST_CASE("run rejects an operator without two operands") {
    Holder variables;
    Rpn rpn;
    rpn.push_back(Unit::createScalar(2));
    rpn.push_back(Unit::createOperator('+'));
    CHECK_THROWS_WITH_AS(Expression{std::move(rpn)}.run(variables),
                         doctest::Contains("Ill-formed RPN"), std::runtime_error);
}

TEST_CASE("run rejects an RPN that leaves more than one value on the stack") {
    Holder variables;
    Rpn rpn;
    rpn.push_back(Unit::createScalar(2));
    rpn.push_back(Unit::createScalar(3));
    CHECK_THROWS_WITH_AS(Expression{std::move(rpn)}.run(variables),
                         doctest::Contains("stack imbalance"), std::runtime_error);
}

TEST_CASE("run rejects a lone function unit") {
    // This is the shape insertUnary produces: a function with no '#' to apply it.
    Holder variables;
    Rpn rpn;
    rpn.push_back(Unit::createFunction("sin"));
    CHECK_THROWS_WITH_AS(Expression{std::move(rpn)}.run(variables),
                         doctest::Contains("Last value is not a number"), std::runtime_error);
}

TEST_CASE("run rejects a plain operator applied to a function unit") {
    Holder variables;
    Rpn rpn;
    rpn.push_back(Unit::createFunction("sin"));
    rpn.push_back(Unit::createScalar(2));
    rpn.push_back(Unit::createOperator('+'));
    CHECK_THROWS_WITH_AS(Expression{std::move(rpn)}.run(variables),
                         doctest::Contains("Can't scalarize"), std::runtime_error);
}

TEST_CASE("run rejects a function application whose left side is not a function") {
    Holder variables;
    Rpn rpn;
    rpn.push_back(Unit::createScalar(2));
    rpn.push_back(Unit::createScalar(3));
    rpn.push_back(Unit::createOperator('#'));
    CHECK_THROWS_AS(Expression{std::move(rpn)}.run(variables), std::bad_variant_access);
}

TEST_CASE("An RPN with an extra operand is rejected wherever the operand sits") {
    // insertUnary splices a function unit at an arbitrary index, so the damage
    // lands in different places; all of them have to fail, none may compute.
    Holder variables;
    variables.setVariable("x", 2);
    for (const std::size_t at : {std::size_t{0}, std::size_t{1}, std::size_t{2}}) {
        Rpn rpn;
        rpn.push_back(Unit::createVariable("x"));
        rpn.push_back(Unit::createVariable("x"));
        rpn.push_back(Unit::createOperator('+'));
        rpn.insert(rpn.begin() + at, Unit::createFunction("sin"));
        CHECK_THROWS_AS(Expression{std::move(rpn)}.run(variables), std::exception);
    }
}


// --- toString ------------------------------------------------------------
// The printed form is the rank-cache key, it feeds the length term of the
// fitness, and it is what the golden snapshots compare. It has to stay exact.

TEST_CASE("toString parenthesizes by precedence") {
    CHECK(Expression("x+x").toString() == "x+x");
    CHECK(Expression("x*2+1").toString() == "x*2+1");
    CHECK(Expression("(x+1)*2").toString() == "(x+1)*2");
    CHECK(Expression("x*(x+1)").toString() == "x*(x+1)");
    CHECK(Expression("(x+1)^2").toString() == "(x+1)^2");
}

TEST_CASE("toString keeps the strict-side parentheses of minus and divide") {
    // a-(b+c) and a/(b*c) change meaning if the parentheses are dropped, while
    // a+(b+c) does not.
    CHECK(Expression("x-(x+1)").toString() == "x-(x+1)");
    CHECK(Expression("x/(x*2)").toString() == "x/(x*2)");
    CHECK(Expression("x+(x+1)").toString() == "x+x+1");
}

TEST_CASE("toString renders a function application as a call") {
    CHECK(Expression("sin(x)").toString() == "sin(x)");
    CHECK(Expression("sin(x+1)").toString() == "sin(x+1)");
    CHECK(Expression("sin(x)*2").toString() == "sin(x)*2");
}

TEST_CASE("toString trims trailing zeros off numbers") {
    const auto printed = [](const double value) {
        Rpn rpn;
        rpn.push_back(Unit::createScalar(value));
        return Expression{std::move(rpn)}.toString();
    };
    CHECK(printed(2.0) == "2");
    CHECK(printed(2.5) == "2.5");
    CHECK(printed(0.25) == "0.25");
    CHECK(printed(1.23456) == "1.2346");
    CHECK(printed(-1.5) == "-1.5");
}

TEST_CASE("toString wraps a negative literal that sits under an operator") {
    // Golden snapshots contain rows like `x-(-4.2227)`, so the wrapping of a
    // negative leaf is observable output, not an internal detail.
    Rpn rpn;
    rpn.push_back(Unit::createVariable("x"));
    rpn.push_back(Unit::createScalar(-4.2227));
    rpn.push_back(Unit::createOperator('-'));
    CHECK(Expression{std::move(rpn)}.toString() == "x-(-4.2227)");
}


// --- parsing -------------------------------------------------------------

TEST_CASE("Parsing and printing round-trip") {
    for (const auto* text : {"x+x", "x*2+1", "(x+1)*2", "sin(x)", "x/(x*2)", "x^2"}) {
        CHECK(Expression(text).toString() == Expression(Expression(text).toString()).toString());
    }
}

TEST_CASE("A unary sign becomes a subtraction from zero") {
    CHECK(EvalWithX("-x", 3) == doctest::Approx(-3));
    CHECK(EvalWithX("-x+1", 3) == doctest::Approx(-2));
    CHECK(EvalWithX("(-x)*2", 3) == doctest::Approx(-6));
}

TEST_CASE("The validator catches what the parser cannot recover from") {
    using Validator = Matematyka::ExpressionValidator;
    CHECK(Validator::Validate("x+1").error == Validator::Error::NoError);
    CHECK(Validator::Validate("(x+1)*2").error == Validator::Error::NoError);
    CHECK(Validator::Validate("(x+1").error == Validator::Error::ParenthesesMismatch);
    CHECK(Validator::Validate("x+1)").error == Validator::Error::ParenthesesMismatch);
    CHECK(Validator::Validate("x**1").error == Validator::Error::ConsecutiveOperators);

    CHECK_THROWS_AS(Expression("(x+1"), std::runtime_error);
}


// --- known defects, recorded so a rewrite cannot change them by accident ---

TEST_CASE("BUG: a binary + or - directly after ')' is parsed as a unary sign") {
    // Operation::IsOperator returns true for '(' and ')', and SimplifyExpresssion
    // uses it to decide whether a sign is unary. A ')' closes a value, so every
    // `)+` and `)-` gets a spurious `0` operand spliced in and the expression is
    // left ill-formed. Any expression of this extremely common shape is silently
    // dead -- ranked 0 -- wherever a user can type one: --expression on the CLI,
    // "Initial expressions" in the GUI, and Expression(toString()) round-trips.
    CHECK(Expression::SimplifyExpresssion("sin(x)+1") == "sin#(x)0+1");
    CHECK(Expression::SimplifyExpresssion("(x+1)-2") == "(x+1)0-2");

    Holder variables;
    variables.setVariable("x", 0.5);
    CHECK_THROWS_AS(Eval("sin(x)+1", variables), std::bad_variant_access);
    CHECK_THROWS_WITH(Eval("(x+1)-2", variables), doctest::Contains("stack imbalance"));
    CHECK_THROWS_WITH(Eval("(x+1)+2", variables), doctest::Contains("stack imbalance"));

    // Writing the same thing with the value first happens to work.
    CHECK(Eval("1+sin(x)", variables) == doctest::Approx(1. + std::sin(0.5)));
    CHECK(EvalWithX("(x+1)*2", 1) == doctest::Approx(4));
}

TEST_CASE("BUG: a unary sign is spliced in without parentheses, so it binds wrong") {
    // `x*-1` becomes `x*0-1`, which is (x*0)-1 = -1 for any x, not -x.
    CHECK(Expression::SimplifyExpresssion("x*-1") == "x*0-1");
    CHECK(EvalWithX("x*-1", 2) == doctest::Approx(-1));   // should be -2
    CHECK(EvalWithX("x^-1", 2) == doctest::Approx(0));    // x^0-1; should be 0.5
}

TEST_CASE("An ill-formed RPN prints as itself, never as a valid expression") {
    // toString used to return stack.top() without checking that exactly one node
    // was left, so leftover operands were silently dropped and `[3, x, 1, +]`
    // came out as "x+1" -- indistinguishable from the real thing. That matters
    // beyond display: the rank cache is keyed on this string, so whichever of the
    // two was ranked first handed the other its rank.
    Rpn good;
    good.push_back(Unit::createVariable("x"));
    good.push_back(Unit::createScalar(1));
    good.push_back(Unit::createOperator('+'));

    Rpn leftover = good;
    leftover.insert(leftover.begin(), Unit::createScalar(3));

    Rpn starved;
    starved.push_back(Unit::createOperator('+'));

    CHECK(Expression{std::move(good)}.toString() == "x+1");
    CHECK(Expression{std::move(leftover)}.toString() == "<ill-formed: 3 x 1 +>");
    CHECK(Expression{std::move(starved)}.toString() == "<ill-formed: +>");
}

TEST_CASE("toString is total: an empty expression prints rather than misbehaving") {
    CHECK(Expression{}.toString() == "<ill-formed:>");
}

TEST_CASE("Left-associativity of ^ is what the parser produces") {
    // Not the mathematical convention: MakeRpn pops while priority(top) >= priority(c).
    CHECK(EvalWithX("2^3^2", 0) == doctest::Approx(64));  // (2^3)^2, not 2^(3^2)
}

// Deliberately untested: VariableHolder::getVariable() dereferences find()
// without comparing against end(), so it is undefined behaviour on a missing
// name rather than a defect with a value worth pinning. Nothing calls it --
// every reachable path goes through getVariableSafe.
