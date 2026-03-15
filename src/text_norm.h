#pragma once
#include <string>
#include <cctype>

inline std::string normalize_spanish_lower(const std::string& s) {
    std::string out;
    out.reserve(s.size());
    for (size_t i = 0; i < s.size(); ++i) {
        unsigned char c = (unsigned char)s[i];
        if (c == 0xC3 && i + 1 < s.size()) {
            unsigned char d = (unsigned char)s[i + 1];
            switch (d) {
                case 0xA1: case 0x81: out.push_back('a'); break; // a/Á
                case 0xA9: case 0x89: out.push_back('e'); break; // e/É
                case 0xAD: case 0x8D: out.push_back('i'); break; // i/Í
                case 0xB3: case 0x93: out.push_back('o'); break; // o/Ó
                case 0xBA: case 0x9A: out.push_back('u'); break; // u/Ú
                case 0xBC: case 0x9C: out.push_back('u'); break; // u/Ü
                case 0xB1: case 0x91: out.push_back('n'); break; // n/Ñ
                default:
                    break;
            }
            i++;
            continue;
        }
        out.push_back((char)std::tolower(c));
    }
    return out;
}
