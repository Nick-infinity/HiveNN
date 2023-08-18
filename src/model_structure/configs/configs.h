#pragma once
#include "utils.h"
#include "types.h"

// TODO : Expand Default Config Parameters for generic structure
class Config
{ /* Keep it for Polymorphism or newer configurations*/
    public:
    CONFIG_TYPE config_type = CONFIG_TYPE::NONE;
};
