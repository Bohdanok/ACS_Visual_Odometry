//
// Created by julfy1 on 3/24/25.
//
#pragma once
#ifndef FREAK_FEATURE_DESCRIPTOR_PARALLEL_H
#define FREAK_FEATURE_DESCRIPTOR_PARALLEL_H

#include <opencv2/core.hpp>
#include <vector>
#include <array>

constexpr size_t KEY_POINTS_PER_TASK = 30;
// #ifndef CORNER_DETECTION
#ifndef FEATURE_EXTRACTION_PARALLEL_GPU_H
struct point {
    int x, y;
};

struct test {
    cv::KeyPoint point1, point2;
};

// #endif



constexpr size_t NUM_POINTS = 43;
constexpr size_t NUM_PAIRS = (NUM_POINTS * (NUM_POINTS - 1)) / 2;



constexpr std::array<point, NUM_POINTS> predefined_point_for_matching = {{
    {33, 0}, {17, -30}, {-17, -30}, {-33, 0}, {-17, 30}, {17, 30},
    {22, 13}, {22, -13}, {0, -26}, {-22, -13}, {-22, 13}, {0, 26},
    {18, 0}, {9, -17}, {-9, -17}, {-18, 0}, {-9, 17}, {9, 17},
    {11, 7}, {11, -7}, {0, -13}, {-11, -7}, {-11, 7}, {0, 13},
    {8, 0}, {4, -8}, {-4, -8}, {-8, 0}, {-4, 8}, {4, 8},
    {5, 3}, {5, -3}, {0, -6}, {-5, -3}, {-5, 3}, {0, 6},
    {4, 0}, {2, -4}, {-2, -4}, {-4, 0}, {-2, 4}, {2, 4},
    {0, 0}
}};

inline std::array<test, NUM_PAIRS> generate_tests() {
    std::array<test, NUM_PAIRS> result{};
    size_t index = 0;

    for (size_t i = 0; i < NUM_POINTS; i++) {
        for (size_t j = i + 1; j < NUM_POINTS; j++) {
            result[index++] = {cv::KeyPoint(cv::Point2f(static_cast<float>(predefined_point_for_matching[i].x), static_cast<float>(predefined_point_for_matching[i].y)), 1.0f),
                   cv::KeyPoint(cv::Point2f(static_cast<float>(predefined_point_for_matching[j].x), static_cast<float>(predefined_point_for_matching[j].y)), 1.0f)};
        }
    }

    return result;
}

static std::array<test, NUM_PAIRS> test_cases = generate_tests();


constexpr std::array<size_t, 512> PATCH_DESCRIPTION_POINTS =
    {
        404,431,818,511,181,52,311,874,774,543,719,230,417,205,11,
        560,149,265,39,306,165,857,250,8,61,15,55,717,44,412,
        592,134,761,695,660,782,625,487,549,516,271,665,762,392,178,
        796,773,31,672,845,548,794,677,654,241,831,225,238,849,83,
        691,484,826,707,122,517,583,731,328,339,571,475,394,472,580,
        381,137,93,380,327,619,729,808,218,213,459,141,806,341,95,
        382,568,124,750,193,749,706,843,79,199,317,329,768,198,100,
        466,613,78,562,783,689,136,838,94,142,164,679,219,419,366,
        418,423,77,89,523,259,683,312,555,20,470,684,123,458,453,833,
        72,113,253,108,313,25,153,648,411,607,618,128,305,232,301,84,
        56,264,371,46,407,360,38,99,176,710,114,578,66,372,653,
        129,359,424,159,821,10,323,393,5,340,891,9,790,47,0,175,346,
        236,26,172,147,574,561,32,294,429,724,755,398,787,288,299,
        769,565,767,722,757,224,465,723,498,467,235,127,802,446,233,
        544,482,800,318,16,532,801,441,554,173,60,530,713,469,30,
        212,630,899,170,266,799,88,49,512,399,23,500,107,524,90,
        194,143,135,192,206,345,148,71,119,101,563,870,158,254,214,
        276,464,332,725,188,385,24,476,40,231,620,171,258,67,109,
        844,244,187,388,701,690,50,7,850,479,48,522,22,154,12,659,
        736,655,577,737,830,811,174,21,237,335,353,234,53,270,62,
        182,45,177,245,812,673,355,556,612,166,204,54,248,365,226,
        242,452,700,685,573,14,842,481,468,781,564,416,179,405,35,
        819,608,624,367,98,643,448,2,460,676,440,240,130,146,184,
        185,430,65,807,377,82,121,708,239,310,138,596,730,575,477,
        851,797,247,27,85,586,307,779,326,494,856,324,827,96,748,
        13,397,125,688,702,92,293,716,277,140,112,4,80,855,839,1,
        413,347,584,493,289,696,19,751,379,76,73,115,6,590,183,734,
        197,483,217,344,330,400,186,243,587,220,780,200,793,246,824,
        41,735,579,81,703,322,760,720,139,480,490,91,814,813,163,
        152,488,763,263,425,410,576,120,319,668,150,160,302,491,515,
        260,145,428,97,251,395,272,252,18,106,358,854,485,144,550,
        131,133,378,68,102,104,58,361,275,209,697,582,338,742,589,
        325,408,229,28,304,191,189,110,126,486,211,547,533,70,215,
        670,249,36,581,389,605,331,518,442,822
    };



constexpr size_t DESCRIPTOR_SIZE = PATCH_DESCRIPTION_POINTS.size();

#endif

class FREAK_Parallel {
public:
    static float compute_orientation(const cv::KeyPoint &point, const cv::Mat& image);
    static void FREAK_feature_description(const std::vector<cv::KeyPoint>& key_points, const cv::Mat blurred_gray_picture, const size_t& starting_key_point_index, std::vector<std::vector<uint8_t>>& descriptor);
    static void FREAK_feature_description_worker(const std::vector<cv::KeyPoint>& key_points, const cv::Mat& blurred_gray_picture, const size_t& starting_key_point_index, std::vector<std::vector<uint8_t>>& descriptor, const size_t& num_of_keypoints, const size_t& KEYPOINTS_PER_TASK = KEY_POINTS_PER_TASK);

};

#endif //FREAK_FEATURE_DESCRIPTOR_PARALLEL_H
