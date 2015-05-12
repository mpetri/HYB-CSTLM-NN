#pragma once

#define ELPP_STL_LOGGING
#define ELPP_NO_DEFAULT_LOG_FILE

#ifndef NDEBUG
#define ELPP_DEBUG_ASSERT_FAILURE
#define ELPP_STACKTRACE_ON_CRASH
#endif

#include "easylogging++.h"
INITIALIZE_EASYLOGGINGPP

struct log {
    inline static void load_default_config(bool print_to_stdout)
    {
        el::Loggers::addFlag(el::LoggingFlag::LogDetailedCrashReason);
        el::Loggers::addFlag(el::LoggingFlag::DisableApplicationAbortOnFatalLog);
        el::Loggers::addFlag(el::LoggingFlag::ColoredTerminalOutput);
        el::Loggers::addFlag(el::LoggingFlag::MultiLoggerSupport);
        el::Loggers::addFlag(el::LoggingFlag::CreateLoggerAutomatically);
        el::Loggers::addFlag(el::LoggingFlag::HierarchicalLogging);
        el::Configurations defaultConf;
        defaultConf.setToDefault();
        defaultConf.setGlobally(el::ConfigurationType::Format, "%datetime %level:  %msg");
        defaultConf.setGlobally(el::ConfigurationType::ToFile, "false");
        if (print_to_stdout)
            defaultConf.setGlobally(el::ConfigurationType::ToStandardOutput, "true");
        else
            defaultConf.setGlobally(el::ConfigurationType::ToStandardOutput, "false");
        el::Loggers::reconfigureAllLoggers(defaultConf);
    }

    inline static void start_log(int argc, const char** argv, bool print_to_stdout = true)
    {
        START_EASYLOGGINGPP(argc, argv);
        load_default_config(print_to_stdout);
    }
};