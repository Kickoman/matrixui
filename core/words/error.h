#pragma once

// One exception hierarchy for the words module.
//
// The rule: environment failures throw (missing file, truncated artifact,
// impossible configuration), and bad *user input* does not -- a word that is
// not in the vocabulary comes back as data on a result struct, so a GUI can
// show it inline instead of raising a dialog.

#include <stdexcept>
#include <string>

namespace Words {

class Error : public std::runtime_error {
public:
    using std::runtime_error::runtime_error;
};

// Missing, unreadable, truncated or wrong-format file.
class IoError : public Error {
public:
    using Error::Error;
};

// The vocabulary itself is unusable (empty, or too large for TWordId).
class VocabularyError : public Error {
public:
    using Error::Error;
};

// A setting that cannot produce a usable run.
class ConfigError : public Error {
public:
    using Error::Error;
};

}  // namespace Words
