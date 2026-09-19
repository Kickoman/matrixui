#pragma once

#include <stdexcept>

namespace Serving {

class Error : public std::runtime_error {
public:
    using std::runtime_error::runtime_error;
};

// The manifest itself: unreadable JSON, a key that does not belong, a path
// that points outside the model directory.
class ManifestError : public Error {
public:
    using Error::Error;
};

// The weights blob: truncated, or not the blob the manifest names.
class IntegrityError : public Error {
public:
    using Error::Error;
};

// The manifest and the network disagree about what the model takes or returns.
class ContractError : public Error {
public:
    using Error::Error;
};

}  // namespace Serving
