#pragma once

#include <string>
#include <unistd.h>
#include <stdlib.h>
#include <iostream>
#include <fstream>

#include <sys/types.h>
#include <sys/stat.h>
#include <unistd.h>

#include "logging.hpp"
#include "timings.hpp"
#include "mem_monitor.hpp"

namespace utils {

bool directory_exists(std::string dir)
{
    struct stat sb;
    const char* pathname = dir.c_str();
    if (stat(pathname, &sb) == 0 && (S_IFDIR & sb.st_mode)) {
        return true;
    }
    return false;
}

bool file_exists(std::string file_name)
{
    std::ifstream in(file_name);
    if (in) {
        in.close();
        return true;
    }
    return false;
}

void create_directory(std::string dir)
{
    if (!directory_exists(dir)) {
        if (mkdir(dir.c_str(), 0777) == -1) {
            LOG(FATAL) << "could not create directory";
        }
    }
}

struct lm_mem_monitor {
    static void start(std::string file) {
        auto& m = instance(file);
        m.event("START");
    }

    static mem_monitor& instance(std::string file = "") {
        static mem_monitor m(file);
        return m;
    }

    static void event(const std::string& ev) {
        auto& m = instance();
        m.event(ev);
    }

    static mem_stat stats() {
        auto& m = instance();
        return m.get_current_stats();
    }
};


}