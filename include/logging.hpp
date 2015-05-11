#pragma once

#define ELPP_STL_LOGGING
#define ELPP_NO_DEFAULT_LOG_FILE

#ifndef NDEBUG
  #define ELPP_DEBUG_ASSERT_FAILURE
  #define ELPP_STACKTRACE_ON_CRASH
#endif

#ifdef LMSDSL_NO_LOGGING
  #define ELPP_DISABLE_LOGS
#endif

#include "easylogging++.h"
INITIALIZE_EASYLOGGINGPP

struct log {
	inline static void load_default_config()
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
		defaultConf.setGlobally(el::ConfigurationType::ToStandardOutput, "true");
		el::Loggers::reconfigureAllLoggers(defaultConf);
	}

    inline static void start_log(int argc,const char** argv)
    {
		START_EASYLOGGINGPP(argc, argv);
		load_default_config();
    }
};