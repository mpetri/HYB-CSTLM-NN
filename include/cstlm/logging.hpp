#pragma once

namespace cstlm {

static bool enable_logging = false;

enum typelog {
    DEBUG,
    INFO,
    WARN,
    FATAL,
    ERROR
};

struct LOG {
public:
    LOG() {}
    LOG(typelog type)
    {
        if (enable_logging)
            operator<<("[" + getLabel(type) + "] ");
    }
    ~LOG()
    {
        if (enable_logging) {
            if (opened) {
                std::cout << std::endl;
            }
        }
    }
    template <class T>
    LOG& operator<<(const T& msg)
    {
        if (enable_logging) {
            std::cout << msg;
            opened = true;
        }
        return *this;
    }

    template <class T>
    LOG& operator<<(const std::vector<T>& vd)
    {
        if (enable_logging) {
            std::cout << "[";
            for (size_t i = 0; i < vd.size() - 1; i++) {
                std::cout << vd[i] << ",";
            }
            std::cout << vd.back() << "]";
            opened = true;
        }
        return *this;
    }

private:
    bool opened = false;
    inline std::string getLabel(typelog type)
    {
        std::string label;
        switch (type) {
        case DEBUG:
            label = "DEBUG";
            break;
        case INFO:
            label = "INFO";
            break;
        case WARN:
            label = "WARN";
            break;
        case FATAL:
            label = "FATAL";
            break;
        case ERROR:
            label = "ERROR";
            break;
        }
        return label;
    }
};
}