# The following five lines of boilerplate have to be in your project's
# CMakeLists in this exact order for cmake to work correctly
cmake_minimum_required(VERSION 3.16)

set(PROJECT_VER "1.4.1")
add_definitions(-DFIRMWARE_VERSION=\"${PROJECT_VER}\")

# Add this line to disable the specific warning
add_compile_options(-Wno-missing-field-initializers)

# App images are signed (sdkconfig.defaults: CONFIG_SECURE_SIGNED_APPS_*) and
# OTA refuses anything not signed with the same key. The private key never
# lives in the repo: it is $STACKCHAN_SIGNING_KEY, else
# ~/.stackchan-keys/firmware_signing_key.pem. Kconfig can't expand ~ or env
# vars, so hand the absolute path over as one more defaults file (later files
# win). Like any default it only seeds a fresh setting: a sdkconfig that
# already names a key keeps it — edit it there (or `idf.py menuconfig` →
# Security features) to move the key.
if(DEFINED ENV{STACKCHAN_SIGNING_KEY})
    set(STACKCHAN_SIGNING_KEY "$ENV{STACKCHAN_SIGNING_KEY}")
else()
    set(STACKCHAN_SIGNING_KEY "$ENV{HOME}/.stackchan-keys/firmware_signing_key.pem")
endif()
file(CONFIGURE OUTPUT "${CMAKE_BINARY_DIR}/sdkconfig.signing"
     CONTENT "CONFIG_SECURE_BOOT_SIGNING_KEY=\"${STACKCHAN_SIGNING_KEY}\"\n")
set(SDKCONFIG_DEFAULTS "sdkconfig.defaults;${CMAKE_BINARY_DIR}/sdkconfig.signing")

include($ENV{IDF_PATH}/tools/cmake/project.cmake)
project(stack-chan)
