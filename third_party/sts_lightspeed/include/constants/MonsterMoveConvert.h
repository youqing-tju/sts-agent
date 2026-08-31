#ifndef STS_LIGHTSPEED_MONSTERMOVECONVERT_H
#define STS_LIGHTSPEED_MONSTERMOVECONVERT_H

#include "sts_common.h"
#include "constants/MonsterIds.h"
#include "constants/MonsterMoves.h"
#include <cassert>
#include <iostream>

namespace sts {
    static MonsterMoveId getMonsterMoveIdFromGameId(MonsterId monster, int id) {
        switch (monster)
        {
        case MonsterId::ACID_SLIME_L:
            switch (id) {
            case 0: return MMID::INVALID;
            case 1: return MMID::ACID_SLIME_L_CORROSIVE_SPIT;
            case 2: return MMID::ACID_SLIME_L_TACKLE;
            case 3: return MMID::ACID_SLIME_L_SPLIT;
            case 4: return MMID::ACID_SLIME_L_LICK;
            default: break;
            }
            break;
        case MonsterId::ACID_SLIME_M:
            switch (id) {
            case 0: return MMID::INVALID;
            case 1: return MMID::ACID_SLIME_M_CORROSIVE_SPIT;
            case 2: return MMID::ACID_SLIME_M_TACKLE;
            case 4: return MMID::ACID_SLIME_M_LICK;
            default: break;
            }
            break;
        case MonsterId::ACID_SLIME_S:
            switch (id) {
            case 0: return MMID::INVALID;
            case 1: return MMID::ACID_SLIME_S_TACKLE;
            case 2: return MMID::ACID_SLIME_S_LICK;
            default: break;
            }
            break;
        case MonsterId::AWAKENED_ONE:
            switch (id) {
            case 0: return MMID::INVALID;
            case 1: return MMID::AWAKENED_ONE_SLASH;
            case 2: return MMID::AWAKENED_ONE_SOUL_STRIKE;
            case 3: return MMID::AWAKENED_ONE_REBIRTH;
            case 5: return MMID::AWAKENED_ONE_DARK_ECHO;
            case 6: return MMID::AWAKENED_ONE_SLUDGE;
            case 8: return MMID::AWAKENED_ONE_TACKLE;
            default: break;
            }
            break;
        case MonsterId::BEAR:
            switch (id) {
            case 0: return MMID::INVALID;
            case 1: return MMID::BEAR_MAUL;
            case 2: return MMID::BEAR_BEAR_HUG;
            case 3: return MMID::BEAR_LUNGE;
            default: break;
            }
            break;
        case MonsterId::POINTY:
            switch (id) {
            case 0: return MMID::INVALID;
            case 1: return MMID::POINTY_ATTACK;
            default: break;
            }
            break;
        case MonsterId::ROMEO:
            switch (id) {
            case 0: return MMID::INVALID;
            case 1: return MMID::ROMEO_AGONIZING_SLASH;
            case 2: return MMID::ROMEO_MOCK;
            case 3: return MMID::ROMEO_CROSS_SLASH;
            default: break;
            }
            break;
        case MonsterId::BOOK_OF_STABBING:
            switch (id) {
            case 0: return MMID::INVALID;
            case 1: return MMID::BOOK_OF_STABBING_MULTI_STAB;
            case 2: return MMID::BOOK_OF_STABBING_SINGLE_STAB;
            default: break;
            }
            break;
        case MonsterId::BRONZE_AUTOMATON:
            switch (id) {
            case 0: return MMID::INVALID;
            case 1: return MMID::BRONZE_AUTOMATON_FLAIL;
            case 2: return MMID::BRONZE_AUTOMATON_HYPER_BEAM;
            case 3: return MMID::BRONZE_AUTOMATON_STUNNED;
            case 4: return MMID::BRONZE_AUTOMATON_SPAWN_ORBS;
            case 5: return MMID::BRONZE_AUTOMATON_BOOST;
            default: break;
            }
            break;
        case MonsterId::BRONZE_ORB:
            switch (id) {
            case 0: return MMID::INVALID;
            case 1: return MMID::BRONZE_ORB_BEAM;
            case 2: return MMID::BRONZE_ORB_SUPPORT_BEAM;
            case 3: return MMID::BRONZE_ORB_STASIS;
            default: break;
            }
            break;
        case MonsterId::BYRD:
            switch (id) {
            case 0: return MMID::INVALID;
            case 1: return MMID::BYRD_PECK;
            case 2: return MMID::BYRD_FLY;
            case 3: return MMID::BYRD_SWOOP;
            case 4: return MMID::BYRD_STUNNED;
            case 5: return MMID::BYRD_HEADBUTT;
            case 6: return MMID::BYRD_CAW;
            default: break;
            }
            break;
        case MonsterId::CENTURION:
            switch (id) {
            case 0: return MMID::INVALID;
            case 1: return MMID::CENTURION_SLASH;
            case 2: return MMID::CENTURION_DEFEND;
            case 3: return MMID::CENTURION_FURY;
            default: break;
            }
            break;
        case MonsterId::THE_CHAMP:
            switch (id) {
            case 0: return MMID::INVALID;
            case 1: return MMID::THE_CHAMP_HEAVY_SLASH;
            case 2: return MMID::THE_CHAMP_DEFENSIVE_STANCE;
            case 3: return MMID::THE_CHAMP_EXECUTE;
            case 4: return MMID::THE_CHAMP_FACE_SLAP;
            case 5: return MMID::THE_CHAMP_ANGER;
            case 6: return MMID::THE_CHAMP_TAUNT;
            case 7: return MMID::THE_CHAMP_EXECUTE;
            default: break;
            }
            break;
        case MonsterId::CHOSEN:
            switch (id) {
            case 0: return MMID::INVALID;
            case 1: return MMID::CHOSEN_ZAP;
            case 2: return MMID::CHOSEN_DRAIN;
            case 3: return MMID::CHOSEN_DEBILITATE;
            case 4: return MMID::CHOSEN_HEX;
            case 5: return MMID::CHOSEN_POKE;
            default: break;
            }
            break;
        case MonsterId::CORRUPT_HEART:
            switch (id) {
            case 0: return MMID::INVALID;
            case 1: return MMID::CORRUPT_HEART_BLOOD_SHOTS;
            case 2: return MMID::CORRUPT_HEART_ECHO;
            case 3: return MMID::CORRUPT_HEART_DEBILITATE;
            case 4: return MMID::CORRUPT_HEART_BUFF;
            default: break;
            }
            break;
        case MonsterId::CULTIST:
            switch (id) {
            case 0: return MMID::INVALID;
            case 1: return MMID::CULTIST_DARK_STRIKE;
            case 3: return MMID::CULTIST_INCANTATION;
            default: break;
            }
            break;
        case MonsterId::DARKLING:
            switch (id) {
            case 0: return MMID::INVALID;
            case 1: return MMID::DARKLING_CHOMP;
            case 2: return MMID::DARKLING_HARDEN;
            case 3: return MMID::DARKLING_NIP;
            case 4: return MMID::DARKLING_REGROW;
            case 5: return MMID::DARKLING_REINCARNATE;
            default: break;
            }
            break;
        case MonsterId::DECA:
            switch (id) {
            case 0: return MMID::DECA_BEAM;
            case 2: return MMID::DECA_SQUARE_OF_PROTECTION;
            default: break;
            }
            break;
        case MonsterId::DONU:
            switch (id) {
            case 0: return MMID::DONU_BEAM;
            case 1: return MMID::DONU_CIRCLE_OF_POWER;
            default: break;
            }
            break;
        case MonsterId::EXPLODER:
            switch (id) {
            case 0: return MMID::INVALID;
            case 1: return MMID::EXPLODER_SLAM;
            default: break;
            }
            break;
        case MonsterId::DAGGER:
            switch (id) {
            case 0: return MMID::INVALID;
            case 1: return MMID::DAGGER_STAB;
            case 2: return MMID::DAGGER_EXPLODE;
            default: break;
            }
            break;
        case MonsterId::REPTOMANCER:
            switch (id) {
            case 0: return MMID::INVALID;
            case 1: return MMID::REPTOMANCER_SNAKE_STRIKE;
            case 2: return MMID::REPTOMANCER_SUMMON;
            case 3: return MMID::REPTOMANCER_BIG_BITE;
            default: break;
            }
            break;
        case MonsterId::FUNGI_BEAST:
            switch (id) {
            case 0: return MMID::INVALID;
            case 1: return MMID::FUNGI_BEAST_BITE;
            case 2: return MMID::FUNGI_BEAST_GROW;
            default: break;
            }
            break;
        case MonsterId::GREEN_LOUSE:
            switch (id) {
            case 0: return MMID::INVALID;
            case 3: return MMID::GREEN_LOUSE_BITE;
            case 4: return MMID::GREEN_LOUSE_SPIT_WEB;
            default: break;
            }
            break;
        case MonsterId::RED_LOUSE:
            switch (id) {
            case 0: return MMID::INVALID;
            case 3: return MMID::RED_LOUSE_BITE;
            case 4: return MMID::RED_LOUSE_GROW;
            default: break;
            }
            break;
        case MonsterId::GIANT_HEAD:
            switch (id) {
            case 0: return MMID::INVALID;
            case 1: return MMID::GIANT_HEAD_GLARE;
            case 2: return MMID::GIANT_HEAD_COUNT;
            case 3: return MMID::GIANT_HEAD_IT_IS_TIME;
            default: break;
            }
            break;
        case MonsterId::FAT_GREMLIN:
            switch (id) {
            case 0: return MMID::INVALID;
            case 2: return MMID::FAT_GREMLIN_SMASH;
            default: break;
            }
            break;
        case MonsterId::GREMLIN_LEADER:
            switch (id) {
            case 0: return MMID::INVALID;
            case 2: return MMID::GREMLIN_LEADER_RALLY;
            case 3: return MMID::GREMLIN_LEADER_ENCOURAGE;
            case 4: return MMID::GREMLIN_LEADER_STAB;
            default: break;
            }
            break;
        case MonsterId::GREMLIN_NOB:
            switch (id) {
            case 0: return MMID::INVALID;
            case 1: return MMID::GREMLIN_NOB_RUSH;
            case 2: return MMID::GREMLIN_NOB_SKULL_BASH;
            case 3: return MMID::GREMLIN_NOB_BELLOW;
            default: break;
            }
            break;
        case MonsterId::SNEAKY_GREMLIN:
            switch (id) {
            case 0: return MMID::INVALID;
            case 1: return MMID::SNEAKY_GREMLIN_PUNCTURE;
            default: break;
            }
            break;
        case MonsterId::SHIELD_GREMLIN:
            switch (id) {
            case 0: return MMID::INVALID;
            case 1: return MMID::SHIELD_GREMLIN_PROTECT;
            case 2: return MMID::SHIELD_GREMLIN_SHIELD_BASH;
            default: break;
            }
            break;
        case MonsterId::MAD_GREMLIN:
            switch (id) {
            case 0: return MMID::INVALID;
            case 1: return MMID::MAD_GREMLIN_SCRATCH;
            default: break;
            }
            break;
        case MonsterId::GREMLIN_WIZARD:
            switch (id) {
            case 0: return MMID::INVALID;
            case 1: return MMID::GREMLIN_WIZARD_CHARGING;
            case 2: return MMID::GREMLIN_WIZARD_ULTIMATE_BLAST;
            default: break;
            }
            break;
        case MonsterId::MYSTIC:
            switch (id) {
            case 0: return MMID::INVALID;
            case 1: return MMID::MYSTIC_ATTACK_DEBUFF;
            case 2: return MMID::MYSTIC_HEAL;
            case 3: return MMID::MYSTIC_BUFF;
            default: break;
            }
            break;
        case MonsterId::HEXAGHOST:
            switch (id) {
            case 0: return MMID::INVALID;
            case 1: return MMID::HEXAGHOST_DIVIDER;
            case 2: return MMID::HEXAGHOST_TACKLE;
            case 3: return MMID::HEXAGHOST_INFLAME;
            case 4: return MMID::HEXAGHOST_SEAR;
            case 5: return MMID::HEXAGHOST_ACTIVATE;
            case 6: return MMID::HEXAGHOST_INFERNO;
            default: break;
            }
            break;
        case MonsterId::JAW_WORM:
            switch (id) {
            case 0: return MMID::INVALID;
            case 1: return MMID::JAW_WORM_CHOMP;
            case 2: return MMID::JAW_WORM_BELLOW;
            case 3: return MMID::JAW_WORM_THRASH;
            default: break;
            }
            break;
        case MonsterId::LAGAVULIN:
            switch (id) {
            case 0: return MMID::INVALID;
            case 1: return MMID::LAGAVULIN_SIPHON_SOUL;
            case 3: return MMID::LAGAVULIN_ATTACK;
            case 4: return MMID::LAGAVULIN_SLEEP;
            case 5: return MMID::LAGAVULIN_SLEEP;
            case 6: return MMID::LAGAVULIN_SLEEP;
            default: break;
            }
            break;
        case MonsterId::LOOTER:
            switch (id) {
            case 0: return MMID::INVALID;
            case 1: return MMID::LOOTER_MUG;
            case 2: return MMID::LOOTER_SMOKE_BOMB;
            case 3: return MMID::LOOTER_ESCAPE;
            case 4: return MMID::LOOTER_LUNGE;
            default: break;
            }
            break;
        case MonsterId::THE_MAW:
            switch (id) {
            case 0: return MMID::INVALID;
            case 2: return MMID::THE_MAW_ROAR;
            case 3: return MMID::THE_MAW_SLAM;
            case 4: return MMID::THE_MAW_DROOL;
            case 5: return MMID::THE_MAW_NOM;
            default: break;
            }
            break;
        case MonsterId::MUGGER:
            switch (id) {
            case 0: return MMID::INVALID;
            case 1: return MMID::MUGGER_MUG;
            case 2: return MMID::MUGGER_SMOKE_BOMB;
            case 3: return MMID::MUGGER_ESCAPE;
            case 4: return MMID::MUGGER_LUNGE;
            default: break;
            }
            break;
        case MonsterId::NEMESIS:
            switch (id) {
            case 0: return MMID::INVALID;
            case 2: return MMID::NEMESIS_ATTACK;
            case 3: return MMID::NEMESIS_DEBUFF;
            case 4: return MMID::NEMESIS_SCYTHE;
            default: break;
            }
            break;
        case MonsterId::ORB_WALKER:
            switch (id) {
            case 0: return MMID::INVALID;
            case 1: return MMID::ORB_WALKER_LASER;
            case 2: return MMID::ORB_WALKER_CLAW;
            default: break;
            }
            break;
        case MonsterId::REPULSOR:
            switch (id) {
            case 0: return MMID::INVALID;
            case 1: return MMID::REPULSOR_REPULSE;
            case 2: return MMID::REPULSOR_BASH;
            default: break;
            }
            break;
        case MonsterId::SENTRY:
            switch (id) {
            case 0: return MMID::INVALID;
            case 3: return MMID::SENTRY_BOLT;
            case 4: return MMID::SENTRY_BEAM;
            default: break;
            }
            break;
        case MonsterId::SPIRE_GROWTH:
            switch (id) {
            case 0: return MMID::INVALID;
            case 1: return MMID::SPIRE_GROWTH_QUICK_TACKLE;
            case 2: return MMID::SPIRE_GROWTH_CONSTRICT;
            case 3: return MMID::SPIRE_GROWTH_SMASH;
            default: break;
            }
            break;
        case MonsterId::SHELLED_PARASITE:
            switch (id) {
            case 0: return MMID::INVALID;
            case 1: return MMID::SHELLED_PARASITE_FELL;
            case 2: return MMID::SHELLED_PARASITE_DOUBLE_STRIKE;
            case 3: return MMID::SHELLED_PARASITE_SUCK;
            case 4: return MMID::SHELLED_PARASITE_STUNNED;
            default: break;
            }
            break;
        case MonsterId::BLUE_SLAVER:
            switch (id) {
            case 0: return MMID::INVALID;
            case 1: return MMID::BLUE_SLAVER_STAB;
            case 4: return MMID::BLUE_SLAVER_RAKE;
            default: break;
            }
            break;
        case MonsterId::TASKMASTER:
            switch (id) {
            case 0: return MMID::INVALID;
            case 2: return MMID::TASKMASTER_SCOURING_WHIP;
            default: break;
            }
            break;
        case MonsterId::RED_SLAVER:
            switch (id) {
            case 0: return MMID::INVALID;
            case 1: return MMID::RED_SLAVER_STAB;
            case 2: return MMID::RED_SLAVER_ENTANGLE;
            case 3: return MMID::RED_SLAVER_SCRAPE;
            default: break;
            }
            break;
        case MonsterId::SLIME_BOSS:
            switch (id) {
            case 0: return MMID::INVALID;
            case 1: return MMID::SLIME_BOSS_SLAM;
            case 2: return MMID::SLIME_BOSS_PREPARING;
            case 3: return MMID::SLIME_BOSS_SPLIT;
            case 4: return MMID::SLIME_BOSS_GOOP_SPRAY;
            default: break;
            }
            break;
        case MonsterId::SNAKE_PLANT:
            switch (id) {
            case 0: return MMID::INVALID;
            case 1: return MMID::SNAKE_PLANT_CHOMP;
            case 2: return MMID::SNAKE_PLANT_ENFEEBLING_SPORES;
            default: break;
            }
            break;
        case MonsterId::SNECKO:
            switch (id) {
            case 0: return MMID::INVALID;
            case 1: return MMID::SNECKO_PERPLEXING_GLARE;
            case 2: return MMID::SNECKO_BITE;
            case 3: return MMID::SNECKO_TAIL_WHIP;
            default: break;
            }
            break;
        case MonsterId::SPHERIC_GUARDIAN:
            switch (id) {
            case 0: return MMID::INVALID;
            case 1: return MMID::SPHERIC_GUARDIAN_SLAM;
            case 2: return MMID::SPHERIC_GUARDIAN_ACTIVATE;
            case 3: return MMID::SPHERIC_GUARDIAN_HARDEN;
            case 4: return MMID::SPHERIC_GUARDIAN_ATTACK_DEBUFF;
            default: break;
            }
            break;
        case MonsterId::SPIKE_SLIME_L:
            switch (id) {
            case 0: return MMID::INVALID;
            case 1: return MMID::SPIKE_SLIME_L_FLAME_TACKLE;
            case 3: return MMID::SPIKE_SLIME_L_SPLIT;
            case 4: return MMID::SPIKE_SLIME_L_LICK;
            default: break;
            }
            break;
        case MonsterId::SPIKE_SLIME_M:
            switch (id) {
            case 0: return MMID::INVALID;
            case 1: return MMID::SPIKE_SLIME_M_FLAME_TACKLE;
            case 4: return MMID::SPIKE_SLIME_M_LICK;
            default: break;
            }
            break;
        case MonsterId::SPIKE_SLIME_S:
            switch (id) {
            case 0: return MMID::INVALID;
            case 1: return MMID::SPIKE_SLIME_S_TACKLE;
            default: break;
            }
            break;
        case MonsterId::SPIKER:
            switch (id) {
            case 0: return MMID::INVALID;
            case 1: return MMID::SPIKER_CUT;
            case 2: return MMID::SPIKER_SPIKE;
            default: break;
            }
            break;
        case MonsterId::THE_COLLECTOR:
            switch (id) {
            case 0: return MMID::INVALID;
            case 1:
            case 5: return MMID::THE_COLLECTOR_SPAWN;
            case 2: return MMID::THE_COLLECTOR_FIREBALL;
            case 3: return MMID::THE_COLLECTOR_BUFF;
            case 4: return MMID::THE_COLLECTOR_MEGA_DEBUFF;
            default: break;
            }
            break;
        case MonsterId::THE_GUARDIAN:
            switch (id) {
            case 0: return MMID::INVALID;
            case 1: return MMID::THE_GUARDIAN_DEFENSIVE_MODE;
            case 2: return MMID::THE_GUARDIAN_FIERCE_BASH;
            case 3: return MMID::THE_GUARDIAN_ROLL_ATTACK;
            case 4: return MMID::THE_GUARDIAN_TWIN_SLAM;
            case 5: return MMID::THE_GUARDIAN_WHIRLWIND;
            case 6: return MMID::THE_GUARDIAN_CHARGING_UP;
            case 7: return MMID::THE_GUARDIAN_VENT_STEAM;
            default: break;
            }
            break;
        case MonsterId::TIME_EATER:
            switch (id) {
            case 0: return MMID::INVALID;
            case 2: return MMID::TIME_EATER_REVERBERATE;
            case 3: return MMID::TIME_EATER_RIPPLE;
            case 4: return MMID::TIME_EATER_HEAD_SLAM;
            case 5: return MMID::TIME_EATER_HASTE;
            default: break;
            }
            break;
        case MonsterId::TORCH_HEAD:
            switch (id) {
            case 0: return MMID::INVALID;
            case 1: return MMID::TORCH_HEAD_TACKLE;
            default: break;
            }
            break;
        case MonsterId::TRANSIENT:
            switch (id) {
            case 0: return MMID::INVALID;
            case 1: return MMID::TRANSIENT_ATTACK;
            default: break;
            }
            break;
        case MonsterId::SPIRE_SHIELD:
            switch (id) {
            case 0: return MMID::INVALID;
            case 1: return MMID::SPIRE_SHIELD_BASH;
            case 2: return MMID::SPIRE_SHIELD_FORTIFY;
            case 3: return MMID::SPIRE_SHIELD_SMASH;
            default: break;
            }
            break;
        case MonsterId::SPIRE_SPEAR:
            switch (id) {
            case 0: return MMID::INVALID;
            case 1: return MMID::SPIRE_SPEAR_BURN_STRIKE;
            case 2: return MMID::SPIRE_SPEAR_PIERCER;
            case 3: return MMID::SPIRE_SPEAR_SKEWER;
            default: break;
            }
            break;
        case MonsterId::WRITHING_MASS:
            switch (id) {
            case 0: return MMID::WRITHING_MASS_STRONG_STRIKE;
            case 1: return MMID::WRITHING_MASS_MULTI_STRIKE;
            case 2: return MMID::WRITHING_MASS_FLAIL;
            case 3: return MMID::WRITHING_MASS_WITHER;
            case 4: return MMID::WRITHING_MASS_IMPLANT;
            default: break;
            }
            break;
        case MonsterId::INVALID:
            return MMID::INVALID;
        default:
        #ifdef sts_print_debug
            std::cerr << "MonsterId" << sts::monsterIdStrings[static_cast<int>(monster)] << " not found in getMonsterMoveIdFromGameId\n";
        #endif
        #ifdef sts_asserts
            assert(false);
        #endif
            break;
        }
        #ifdef sts_print_debug
            std::cerr << "Unknown move" << id << " for MonsterId" << sts::monsterIdStrings[static_cast<int>(monster)] << " in getMonsterMoveIdFromGameId\n";
        #endif
        #ifdef sts_asserts

            assert(false);
        #endif
        return MMID::INVALID;
    }
}

#endif //STS_LIGHTSPEED_MONSTERMOVECONVERT_H
