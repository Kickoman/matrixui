#include "core/functions/applier.h"
#include "core/lib/tabulator.h"
#include <cassert>


namespace Genetizer {

namespace {

std::size_t argCount(const Expression::TUnit& u) {
    switch (u.getType()) {
        case Matematyka::rpn::UnitType::Number:
        case Matematyka::rpn::UnitType::Variable:
        case Matematyka::rpn::UnitType::Function:
            return 0;
        case Matematyka::rpn::UnitType::Operator:
            return 2;
    }
    return 0;
}

std::size_t subtreeBegin(const Expression::TRpn& rpn, std::size_t root) {
    long long need = 1;
    for (std::size_t i = root; ; --i) {
        need += static_cast<long long>(argCount(rpn[i])) - 1;
        if (need == 0) {
            return i;
        }
    }
}

std::mt19937& rng() {
    static thread_local std::mt19937 gen{std::random_device{}()};
    return gen;
}

std::size_t randIndex(std::size_t n) {
    return std::uniform_int_distribution<std::size_t>{0, n - 1}(rng());
}

bool coinFlip() {
    return std::uniform_int_distribution<int>{0, 1}(rng()) == 1;
}

char randomOperator(const MutationConfig& cfg) {
    return cfg.operators[randIndex(cfg.operators.size())];
}

std::string_view randomFunction(const MutationConfig&) {
    static auto& functions = Matematyka::rpn::FUNCTIONS_AVAILABLE<double>;
    return functions[randIndex(functions.size())].name;
}

std::string randomVarId(const MutationConfig& cfg) {
    return cfg.variables.at(randIndex(cfg.variables.size()));
}

double randomScalar (const MutationConfig& c) {
    return std::uniform_real_distribution<double>{-c.scalarRange, c.scalarRange}(rng());
}

Expression::TUnit randomOperand(const MutationConfig& c) {
    return coinFlip() ? Expression::TUnit::createScalar(randomScalar(c)) : Expression::TUnit::createVariable(randomVarId(c));
}

std::size_t randomRoot(const Expression::TRpn& rpn) {
    std::size_t eligible = 0;
    for (const auto& u : rpn) {
        if (u.getType() != Matematyka::rpn::UnitType::Function) {
            ++eligible;
        }
    }
    assert(eligible > 0);
    std::size_t k = randIndex(eligible);
    for (std::size_t i = 0; i < rpn.size(); ++i) {
        if (rpn[i].getType() == Matematyka::rpn::UnitType::Function) {
            continue;
        }
        if (k == 0) {
            return i;
        }
        --k;
    }
    return rpn.size() - 1;
}


double unitInterval() {
    return std::uniform_real_distribution<double>{0.0, 1.0}(rng());
}

void growTree(Expression::TRpn& out, const MutationConfig& c,
              std::size_t depthLeft, bool full) {
    const bool makeLeaf = (depthLeft == 0) ? true : full ? false : (unitInterval() < 0.30);

    if (makeLeaf) {
        out.push_back(randomOperand(c));
        return;
    }

    if (unitInterval() < 0.25) {
        out.push_back(Expression::TUnit::createFunction(randomFunction(c)));
        growTree(out, c, depthLeft - 1, full);
        out.push_back(Expression::TUnit::createOperator('#'));
    } else {
        growTree(out, c, depthLeft - 1, full);
        growTree(out, c, depthLeft - 1, full);
        out.push_back(Expression::TUnit::createOperator(randomOperator(c)));
    }
}

Expression randomExpression(const MutationConfig& c, std::size_t maxDepth, bool full) {
    Expression::TRpn rpn;
    growTree(rpn, c, maxDepth, full);
    return Expression{std::move(rpn)};
}


void pointMutate(Expression::TRpn& rpn, const MutationConfig& c) {
    const std::size_t i = randIndex(rpn.size());
    switch (rpn[i].getType()) {
        case Matematyka::rpn::UnitType::Number:
        case Matematyka::rpn::UnitType::Variable:
            rpn[i] = randomOperand(c);
            break;
        case Matematyka::rpn::UnitType::Function:
            rpn[i] = Expression::TUnit::createFunction(randomFunction(c));
            break;
        case Matematyka::rpn::UnitType::Operator:
            rpn[i] = Expression::TUnit::createOperator(randomOperator(c));
            break;
    }
}

void jitterConstant(Expression::TRpn& rpn, const MutationConfig& c) {
    std::vector<std::size_t> nums;
    for (std::size_t i = 0; i < rpn.size(); ++i) {
        if (rpn[i].getType() == Matematyka::rpn::UnitType::Number) {
            nums.push_back(i);
        }
    }
    if (nums.empty()) {
        pointMutate(rpn, c);
        return;
    }

    const std::size_t i = nums[randIndex(nums.size())];
    const double old = std::get<double>(rpn[i].getValue());
    std::normal_distribution<double> step{0.0, 0.1 * std::abs(old) + 0.1};
    rpn[i] = Expression::TUnit::createScalar(old + step(rng()));
}

void insertUnary(Expression::TRpn& rpn, const MutationConfig& c) {
    const std::size_t root = randIndex(rpn.size());
    rpn.insert(rpn.begin() + root + 1, Expression::TUnit::createFunction(randomFunction(c)));
}

void wrapBinary(Expression::TRpn& rpn, const MutationConfig& c) {
    const std::size_t root = randIndex(rpn.size());
    rpn.insert(rpn.begin() + root + 1, randomOperand(c));
    rpn.insert(rpn.begin() + root + 2, Expression::TUnit::createOperator(randomOperator(c)));
}


} // namespace


void FunctionGenetizerApplier::SeedThreadRng(const std::uint64_t seed) {
    rng().seed(seed);
}

void FunctionGenetizerApplier::resetExpected() {
    expectedEntries.resize(0);
    mutationConfig = {};
    knownVariables.clear();
}

void FunctionGenetizerApplier::addExpected(std::vector<Variable>&& variables, const double result) {
    for (const auto &var : variables) {
        if (!knownVariables.contains(var.name)) {
            mutationConfig.variables.push_back(var.name);
            knownVariables.insert(var.name);
        }
    }
    expectedEntries.emplace_back(Entry{
        .variables = std::move(variables),
        .expectedResult = result,
    });
}

void FunctionGenetizerApplier::setMutationOptions(std::vector<char> operators, const TScalar scalarRange) {
    mutationConfig.operators = std::move(operators);
    mutationConfig.scalarRange = scalarRange;
}

OrganismInfo FunctionGenetizerApplier::makeRandomOrganism(const std::size_t maxDepth, const bool full) const {
    return OrganismInfo{
        .epochOfBirth = 0,
        .expression   = randomExpression(mutationConfig, maxDepth, full),
    };
}

void FunctionGenetizerApplier::seedRandom(FunctionGenetizer& genetizer, const std::size_t count, const std::size_t maxDepth) const {
    for (std::size_t i = 0; i < count; ++i) {
        const std::size_t depth = 1 + (i % maxDepth);
        const bool full = (i % 2 == 0);
        genetizer.addOrganism(makeRandomOrganism(depth, full));
    }
}

std::string FunctionGenetizerApplier::PrintWorld(const FunctionGenetizer::TWorld &world, std::size_t top) {
    if (top == 0) {
        top = world.size();
    }
    top = std::min(top, world.size());
    if (top == 0) {
        return "<empty world>\n";
    }
    tabs::Tabulator tabulator;
    tabulator.addHeader() << "Birth" << "Expression" << "Rank";
    for (std::size_t i = 0; i < top; ++i) {
        const auto& org = world[i];
        tabulator.addRow()
            << org.organism.epochOfBirth
            << org.organism.expression.toString()
            << org.rank
        ;
    }
    return tabulator.tabulate();
}

double FunctionGenetizerApplier::rankOrganism(const OrganismInfo& organism) {
    const auto& expression = organism.expression;
    const auto& readable = organism.getPresentation();
    const auto& rpn = expression.getRpn();

    if (auto rank = rankCache.get(readable); rank.has_value()) {
        return *rank;
    }

    if (!std::any_of(expectedEntries.cbegin(), expectedEntries.cend(), [&rpn](const Entry& entry) -> bool {
        const auto& variables = entry.variables;
        return std::any_of(variables.cbegin(), variables.cend(), [&rpn](const Variable& var) -> bool {
            return std::find_if(rpn.cbegin(), rpn.cend(), [&var](const Expression::TUnit& unit) -> bool {
                return unit.getType() == Matematyka::rpn::UnitType::Variable && unit.toString() == var.name;
            }) != rpn.cend();
        });
    })) {
        constexpr auto res = 0.00001;
        rankCache.put(readable, res);
        return res;
    }

    try {
        long double error = 0;
        for (const auto& entry : expectedEntries) {
            for (const auto& var : entry.variables) {
                vars.setVariable(var.name, var.value);
            }
            const auto result = expression.run(vars);
            if (std::isinf(result) || std::isnan(result)) {
                error += 100;
            } else {
                error += std::abs(result - entry.expectedResult);
            }
        }

        const auto complexity = rpn.size();
        const auto complexityFitness = 1. / (1. + complexity / 50.);
        const auto expressionFitness = 1. / (1. + readable.size() / 50.);
        const auto accuracyFitness = 1. / (1. + error);

        constexpr auto accWeightRaw = 0.98;
        constexpr auto comWeightRaw = 0.7;
        constexpr auto expWeightRaw = 0.0001;
        constexpr auto sumWeight = accWeightRaw + comWeightRaw + expWeightRaw;
        constexpr auto accWeight = accWeightRaw / sumWeight;
        constexpr auto comWeight = comWeightRaw / sumWeight;
        constexpr auto expWeight = expWeightRaw / sumWeight;

        const auto res = accWeight * accuracyFitness + comWeight * complexityFitness + expWeight * expressionFitness;
        rankCache.put(readable, res);
        return res;
    } catch (...) {
        rankCache.put(readable, 0);
        return 0.;
    }
}

OrganismInfo FunctionGenetizerApplier::crossoverFunction(const OrganismInfo& mother, const OrganismInfo& father) {
    const auto& motherRpn = mother.expression.getRpn();
    const auto& fatherRpn = father.expression.getRpn();

    const std::size_t mRoot  = randomRoot(motherRpn);
    const std::size_t fRoot  = randomRoot(fatherRpn);
    const std::size_t mBegin = subtreeBegin(motherRpn, mRoot);
    const std::size_t fBegin = subtreeBegin(fatherRpn, fRoot);

    std::decay_t<decltype(motherRpn)> childRpn;
    childRpn.reserve(motherRpn.size() - (mRoot - mBegin + 1) + (fRoot - fBegin + 1));

    childRpn.insert(childRpn.end(), motherRpn.begin(), motherRpn.begin() + mBegin);
    childRpn.insert(childRpn.end(), fatherRpn.begin() + fBegin, fatherRpn.begin() + fRoot + 1);
    childRpn.insert(childRpn.end(), motherRpn.begin() + mRoot + 1, motherRpn.end());

    return OrganismInfo{
        .epochOfBirth = std::max(mother.epochOfBirth, father.epochOfBirth) + 1,
        .expression   = Expression{std::move(childRpn)},
    };
}

void FunctionGenetizerApplier::mutateFunction(OrganismInfo& organism) {
    auto& rpn = organism.expression.getRpnMutable();
    if (rpn.empty()) {
        return;
    }

    enum Kind { Jitter, Point, InsertUnary, WrapBinary };
    static constexpr std::array<Kind, 10> bag{
        Jitter, Jitter, Jitter, Point, Point, Point, Point,
        InsertUnary, InsertUnary, WrapBinary,
    };
    switch (bag[randIndex(bag.size())]) {
        case Jitter:
            jitterConstant(rpn, mutationConfig);
            break;
        case Point:
            pointMutate(rpn, mutationConfig);
            break;
        case InsertUnary:
            insertUnary(rpn, mutationConfig);
            break;
        case WrapBinary:
            wrapBinary(rpn, mutationConfig);
            break;
    }
}

}
