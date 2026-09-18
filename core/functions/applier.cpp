#include "core/functions/applier.h"
#include "core/lib/tabulator.h"

#include <cassert>
#include <string_view>
#include <unordered_map>


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

std::size_t functionAppliedAt(const Expression::TRpn& rpn, const std::size_t application) {
    return subtreeBegin(rpn, application - 1) - 1;
}

std::mt19937& rng() {
    static thread_local std::mt19937 gen{std::random_device{}()};
    return gen;
}

constexpr long double kInvalidResultPenalty = 100;

std::size_t randIndex(std::size_t n) {
    return std::uniform_int_distribution<std::size_t>{0, n - 1}(rng());
}

bool coinFlip() {
    return std::uniform_int_distribution<int>{0, 1}(rng()) == 1;
}

char randomOperator(const MutationConfig& cfg) {
    return cfg.operators[randIndex(cfg.operators.size())];
}

std::string_view randomFunction(const MutationConfig& cfg) {
    assert(!cfg.functions.empty());
    return cfg.functions[randIndex(cfg.functions.size())];
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

    if (!c.functions.empty() && unitInterval() < 0.25) {
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


void wrapBinary(Expression::TRpn& rpn, const MutationConfig& c) {
    const std::size_t root = randomRoot(rpn);
    rpn.insert(rpn.begin() + root + 1, randomOperand(c));
    rpn.insert(rpn.begin() + root + 2, Expression::TUnit::createOperator(randomOperator(c)));
}

void pointMutate(Expression::TRpn& rpn, const MutationConfig& c) {
    const std::size_t i = randIndex(rpn.size());
    switch (rpn[i].getType()) {
        case Matematyka::rpn::UnitType::Number:
        case Matematyka::rpn::UnitType::Variable:
            rpn[i] = randomOperand(c);
            break;
        case Matematyka::rpn::UnitType::Function:
            if (c.functions.empty()) {
                wrapBinary(rpn, c);
                break;
            }
            rpn[i] = Expression::TUnit::createFunction(randomFunction(c));
            break;
        case Matematyka::rpn::UnitType::Operator:
            if (rpn[i].getOperation().getType()
                == Matematyka::rpn::OperationType::FunctionApplication) {
                if (c.functions.empty()) {
                    wrapBinary(rpn, c);
                    break;
                }
                rpn[functionAppliedAt(rpn, i)] =
                    Expression::TUnit::createFunction(randomFunction(c));
                break;
            }
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
    assert(!c.functions.empty());
    const std::size_t root = randomRoot(rpn);
    const std::size_t begin = subtreeBegin(rpn, root);
    rpn.insert(rpn.begin() + root + 1, Expression::TUnit::createOperator('#'));
    rpn.insert(rpn.begin() + begin, Expression::TUnit::createFunction(randomFunction(c)));
}


} // namespace


void FunctionGenetizerApplier::SeedThreadRng(const std::uint64_t seed) {
    rng().seed(seed);
}

void FunctionGenetizerApplier::resetExpected() {
    expectedEntries.resize(0);
    mutationConfig = {};
    knownVariables.clear();
    variableSlots.clear();
    slotAssignments.clear();
    flatEntries.clear();
    slotValues.assign(1, TScalar{});
    expectedMagnitudeSum = 0;
    errorScale = 1.;
    rankCache.clear();
}

std::uint32_t FunctionGenetizerApplier::slotOf(const std::string& name) const {
    const auto found = variableSlots.find(name);
    return found == variableSlots.end() ? kZeroSlot : found->second;
}

void FunctionGenetizerApplier::addExpected(std::vector<Variable>&& variables, const double result) {
    for (const auto &var : variables) {
        if (!knownVariables.contains(var.name)) {
            mutationConfig.variables.push_back(var.name);
            knownVariables.insert(var.name);
            variableSlots.emplace(var.name, static_cast<std::uint32_t>(slotValues.size()));
            slotValues.push_back(TScalar{});
        }
    }

    flatEntries.push_back(FlatEntry{
        .firstAssignment = static_cast<std::uint32_t>(slotAssignments.size()),
        .assignmentCount = static_cast<std::uint32_t>(variables.size()),
        .expectedResult = result,
    });
    for (const auto& var : variables) {
        slotAssignments.push_back(SlotAssignment{
            .slot = slotOf(var.name),
            .value = var.value,
        });
    }

    expectedEntries.emplace_back(Entry{
        .variables = std::move(variables),
        .expectedResult = result,
    });

    expectedMagnitudeSum += std::abs(result);
    errorScale = std::max(1., expectedMagnitudeSum / static_cast<TScalar>(expectedEntries.size()));
    rankCache.clear();
}

void FunctionGenetizerApplier::setMutationOptions(
        std::vector<char> operators, std::vector<std::string> functions,
        const TScalar scalarRange) {
    mutationConfig.operators = std::move(operators);
    mutationConfig.functions = std::move(functions);
    mutationConfig.scalarRange = scalarRange;
}

void FunctionGenetizerApplier::setFitnessOptions(const FitnessConfig& fitness) {
    fitnessConfig = fitness;
    rankCache.clear();
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

DistinctWorld FunctionGenetizerApplier::CollectDistinct(const FunctionGenetizer::TWorld& world, std::size_t top) {
    DistinctWorld result;
    result.totalCount = world.size();

    struct Key {
        std::string_view text;
        std::size_t hash;
    };
    struct KeyHash {
        std::size_t operator()(const Key& key) const { return key.hash; }
    };
    struct KeyEqual {
        bool operator()(const Key& left, const Key& right) const {
            return left.hash == right.hash && left.text == right.text;
        }
    };

    constexpr std::size_t kBeyondTop = static_cast<std::size_t>(-1);

    std::unordered_map<Key, std::size_t, KeyHash, KeyEqual> seen;
    seen.reserve(world.size());

    std::size_t uniqueCount = 0;
    for (const auto& info : world) {
        const Key key{info.organism.getPresentation(), info.organism.getPresentationHash()};
        const auto wanted = (top == 0 || result.rows.size() < top) ? result.rows.size() : kBeyondTop;
        const auto [it, inserted] = seen.try_emplace(key, wanted);
        if (inserted) {
            ++uniqueCount;
            if (wanted != kBeyondTop) {
                result.rows.push_back(DistinctRow{&info, info.organism.epochOfBirth, 1});
            }
        } else if (it->second != kBeyondTop) {
            auto& row = result.rows[it->second];
            ++row.copies;
            // Clones tie on rank, and std::sort is not stable, so the earliest
            // birth in the group is the one reproducible answer to "since when".
            row.birth = std::min(row.birth, info.organism.epochOfBirth);
        }
    }

    result.uniqueCount = uniqueCount;
    return result;
}

std::string FunctionGenetizerApplier::PrintWorld(const FunctionGenetizer::TWorld &world, std::size_t top) {
    if (world.empty()) {
        return "<empty world>\n";
    }

    const auto distinct = CollectDistinct(world, top);

    tabs::Tabulator tabulator;
    tabulator.addHeader() << "Birth" << "Expression" << "Rank" << "Copies";
    for (const auto& row : distinct.rows) {
        tabulator.addRow()
            << row.birth
            << row.representative->organism.getPresentation()
            << row.representative->rank
            << row.copies;
    }
    return "unique " + std::to_string(distinct.uniqueCount) + " / " + std::to_string(distinct.totalCount)
        + "\n" + tabulator.tabulate();
}

double FunctionGenetizerApplier::rankOrganism(const OrganismInfo& organism) {
    const auto& rpn = organism.expression.getRpn();

    if (auto rank = rankCache.get(rpn); rank.has_value()) {
        return *rank;
    }

    const auto mentionsKnownVariable = std::any_of(
        rpn.cbegin(), rpn.cend(), [this](const Expression::TUnit& unit) -> bool {
            return unit.getType() == Matematyka::rpn::UnitType::Variable
                && knownVariables.contains(unit.getVariable());
        });
    if (!mentionsKnownVariable) {
        constexpr auto res = 0.00001;
        rankCache.put(rpn, res);
        return res;
    }

    if (program.compile(rpn, [this](const std::string& name) { return slotOf(name); })
            != Matematyka::CompiledExpression<TScalar>::Status::Ok) {
        rankCache.put(rpn, 0.);
        return 0.;
    }
    if (evaluationStack.size() < program.getStackDepth()) {
        evaluationStack.resize(program.getStackDepth());
    }

    {
        long double error = 0;
        for (const auto& entry : flatEntries) {
            for (std::uint32_t i = 0; i < entry.assignmentCount; ++i) {
                const auto& assignment = slotAssignments[entry.firstAssignment + i];
                slotValues[assignment.slot] = assignment.value;
            }
            const auto result = program.eval(slotValues.data(), evaluationStack.data());
            if (std::isinf(result) || std::isnan(result)) {
                error += kInvalidResultPenalty;
            } else {
                error += std::abs(result - entry.expectedResult) / errorScale;
            }
        }

        const auto complexity = rpn.size();
        const auto complexityFitness = 1. / (1. + complexity / 50.);
        const auto expressionFitness = 1. / (1. + organism.getPresentation().size() / 50.);
        const auto accuracyFitness = 1. / (1. + static_cast<double>(error) / expectedEntries.size());

        const auto accWeightRaw = fitnessConfig.accuracyWeight;
        const auto comWeightRaw = fitnessConfig.complexityWeight;
        const auto expWeightRaw = fitnessConfig.lengthWeight;
        const auto sumWeight = accWeightRaw + comWeightRaw + expWeightRaw;
        const auto accWeight = accWeightRaw / sumWeight;
        const auto comWeight = comWeightRaw / sumWeight;
        const auto expWeight = expWeightRaw / sumWeight;

        const auto res = accWeight * accuracyFitness + comWeight * complexityFitness + expWeight * expressionFitness;
        rankCache.put(rpn, res);
        return res;
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
            if (!mutationConfig.functions.empty()) {
                insertUnary(rpn, mutationConfig);
                break;
            }
            [[fallthrough]];
        case WrapBinary:
            wrapBinary(rpn, mutationConfig);
            break;
    }
}

}
