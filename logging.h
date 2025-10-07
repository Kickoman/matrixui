#pragma once

#include <iostream>


namespace Logging
{

class Log
{
public:
    Log() = default;
    ~Log() {
        std::cout << std::endl;
    }

    template<typename T>
    Log& operator<<(const T& text) {
        std::cout << text;
        return *this;
    }
};

}

#define LOG Log()
