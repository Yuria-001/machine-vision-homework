#include <opencv2/opencv.hpp>
#include <opencv2/core/utils/logger.hpp>

#ifndef NOMINMAX
#define NOMINMAX
#endif
#define WIN32_LEAN_AND_MEAN
#include <windows.h>
#ifdef _MSC_VER
#pragma execution_character_set("utf-8")
#endif

#include <algorithm>
#include <iostream>
#include <string>
#include <vector>

using namespace cv;

int main(int argc, char** argv)
{
    SetConsoleOutputCP(CP_UTF8);
    utils::logging::setLogLevel(utils::logging::LOG_LEVEL_SILENT);

    // 命令行传参优先，否则使用默认图片路径
    std::string imagePath = argc > 1 ? argv[1] : "D:/test3.png";

    Mat image = imread(imagePath, IMREAD_COLOR);
    if (image.empty())
    {
        std::cout << "读不到图片：" << imagePath << std::endl;
        return -1;
    }

    // 1. 灰度化和中值滤波
    Mat gray;
    Mat blurred;
    cvtColor(image, gray, COLOR_BGR2GRAY);
    medianBlur(gray, blurred, 5);

    // 2. 大津法二值化，令目标为白色、背景为黑色
    Mat binary;
    double otsuValue = threshold(
        blurred, binary, 0, 255, THRESH_BINARY | THRESH_OTSU);

    // 根据图像整体亮度判断目标比背景亮还是暗
    double meanGray = mean(blurred)[0];
    if (meanGray > otsuValue)
    {
        threshold(blurred, binary, 0, 255, THRESH_BINARY_INV | THRESH_OTSU);
    }

    // 闭运算连接目标内部的小间断
    Mat kernel = getStructuringElement(MORPH_RECT, Size(3, 3));
    morphologyEx(binary, binary, MORPH_CLOSE, kernel);

    // 3. 连通域标记，获得每个区域的宽、高、面积和质心
    Mat labels;
    Mat stats;
    Mat centroids;

    int labelCount = connectedComponentsWithStats(
        binary, labels, stats, centroids, 8, CV_32S);

    // 原程序的半径范围为15～60，对应直径范围为30～120
    int minCircleSize = 30;
    int maxCircleSize = 120;
    int minArea = 200;

    // 圆形的外接矩形应接近正方形，宽和高越接近，该值越接近1
    double minWidthHeightRatio = 0.8;

    std::vector<Vec3f> circles;

    for (int id = 1; id < labelCount; id++)
    {
        int w = stats.at<int>(id, CC_STAT_WIDTH);
        int h = stats.at<int>(id, CC_STAT_HEIGHT);
        int area = stats.at<int>(id, CC_STAT_AREA);

        bool sizeOK = w >= minCircleSize && w <= maxCircleSize &&
            h >= minCircleSize && h <= maxCircleSize;

        double widthHeightRatio =
            static_cast<double>(std::min(w, h)) / std::max(w, h);

        bool shapeOK = widthHeightRatio >= minWidthHeightRatio;

        // 面积、大小和宽高特征都符合时，认为该区域是圆
        if (area >= minArea && sizeOK && shapeOK)
        {
            float cx = static_cast<float>(centroids.at<double>(id, 0));
            float cy = static_cast<float>(centroids.at<double>(id, 1));
            float radius = static_cast<float>((w + h) / 4.0);

            circles.push_back(Vec3f(cx, cy, radius));
        }
    }

    // 4. 按从上到下、从左到右的顺序编号
    std::sort(
        circles.begin(),
        circles.end(),
        [](const Vec3f& a, const Vec3f& b) {
            if (a[1] != b[1])
                return a[1] < b[1];
            return a[0] < b[0];
        });

    // 5. 绘制圆轮廓、圆心和序号
    Mat result = image.clone();

    for (size_t i = 0; i < circles.size(); ++i)
    {
        int x = cvRound(circles[i][0]);
        int y = cvRound(circles[i][1]);
        int radius = cvRound(circles[i][2]);

        circle(result, Point(x, y), radius, Scalar(0, 0, 255), 2);
        circle(result, Point(x, y), 3, Scalar(255, 255, 255), -1);

        putText(
            result,
            std::to_string(i + 1),
            Point(std::max(10, x - 10),
                std::max(30, y - radius - 10)),
            FONT_HERSHEY_SIMPLEX,
            0.8,
            Scalar(0, 255, 0),
            2);

        std::cout << "圆 " << i + 1
            << "：圆心 = (" << x << ", " << y
            << ")，半径 = " << radius << std::endl;
    }

    // 6. 输出数量并保存、显示结果
    std::string countText =
        "Circle count: " + std::to_string(circles.size());

    putText(
        result,
        countText,
        Point(20, 40),
        FONT_HERSHEY_SIMPLEX,
        1.0,
        Scalar(0, 255, 0),
        2);

    std::cout << "\n识别到的圆数量 = "
        << circles.size() << std::endl;

    imwrite("circle_result.png", result);
    imshow("Circle detection", result);
    waitKey(0);

    return 0;
}