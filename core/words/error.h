#pragma once

#include <stdexcept>
#include <string>

namespace Words {

class Error : public std::runtime_error {
public:
    using std::runtime_error::runtime_error;
};

class IoError : public Error {
public:
    using Error::Error;
};

class VocabularyError : public Error {
public:
    using Error::Error;
};

class ConfigError : public Error {
public:
    using Error::Error;
};

}  // namespace Words
