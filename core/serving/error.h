#pragma once

#include <stdexcept>

namespace Serving {

class Error : public std::runtime_error {
public:
    using std::runtime_error::runtime_error;
};

class ManifestError : public Error {
public:
    using Error::Error;
};

class IntegrityError : public Error {
public:
    using Error::Error;
};

class ContractError : public Error {
public:
    using Error::Error;
};

}  // namespace Serving
