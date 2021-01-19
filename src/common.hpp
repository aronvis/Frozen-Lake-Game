//
// Created by Chi Zhang on 10/22/19.
//

#ifndef FROZEN_LAKE_COMMON_HPP
#define FROZEN_LAKE_COMMON_HPP

#include <cstring>

#ifdef __APPLE__
#define __FILENAME__ (strrchr(__FILE__, '/') ? strrchr(__FILE__, '/') + 1 : __FILE__)
#elif __MINGW32__
#define __FILENAME__ (strrchr(__FILE__, '\\') ? strrchr(__FILE__, '\\') + 1 : __FILE__)
#else
#define __FILENAME__ (strrchr(__FILE__, '/') ? strrchr(__FILE__, '/') + 1 : __FILE__)
#endif

#define MSG(x) std::cout << __FILENAME__ << ':' << __LINE__ << ':' << __func__ << "() : " << x << std::endl

#endif //FROZEN_LAKE_COMMON_HPP
