//
// Created by gamerpuppy on 6/24/2021.
//

#ifndef STS_LIGHTSPEED_CHARACTERCLASSES_H
#define STS_LIGHTSPEED_CHARACTERCLASSES_H

#include <cstdint>
#include <string>

namespace sts {

    enum class CharacterClass : std::uint8_t {
        IRONCLAD=0,
        SILENT,
        DEFECT,
        WATCHER,
        INVALID,
    };

    static constexpr const char* characterClassNames[] = {"Ironclad","Silent","Defect","Watcher"};
    static constexpr const char* characterClassEnumNames[] = {"IRONCLAD","SILENT","DEFECT","WATCHER"};

    static const char* getCharacterClassName(CharacterClass cc) {
        return characterClassNames[static_cast<int>(cc)];
    }

    // class names as sent by the Communication Mod, e.g. "IRONCLAD", "THE_SILENT", "DEFECT", "WATCHER"
    static CharacterClass getCharacterFromCommModName(const std::string &name) {
        if (name == "IRONCLAD") {
            return CharacterClass::IRONCLAD;
        } else if (name == "THE_SILENT") {
            return CharacterClass::SILENT;
        } else if (name == "DEFECT") {
            return CharacterClass::DEFECT;
        } else if (name == "WATCHER") {
            return CharacterClass::WATCHER;
        } else {
            return CharacterClass::INVALID;
        }
    }

}

#endif //STS_LIGHTSPEED_CHARACTERCLASSES_H
