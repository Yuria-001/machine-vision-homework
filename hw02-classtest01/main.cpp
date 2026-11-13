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
#include <iomanip>
#include <iostream>
#include <string>
#include <vector>

using namespace cv;

int main(int argc, char** argv)
{
    SetConsoleOutputCP(CP_UTF8);
    utils::logging::setLogLevel(utils::logging::LOG_LEVEL_SILENT);

    // 命令行传参优先，否则使用默认路径
    std::string imagePath = argc > 1 ? argv[1] : "D:/paper_clip.png";

    // 1. 读取灰度图像
    Mat srcMat = imread(imagePath, IMREAD_GRAYSCALE);
    if (srcMat.empty())
    {
        std::cout << "读不到图片：" << imagePath << std::endl;
        return -1;
    }

    int height = srcMat.rows;
    int width = srcMat.cols;

    std::cout << "输入图像：" << imagePath
        << "  (" << width << " x " << height << ")" << std::endl;

    // 2. 大津法自动计算阈值，反向二值化
    Mat binaryMat;
    double otsuValue = threshold(
        srcMat, binaryMat, 0, 255, THRESH_BINARY_INV | THRESH_OTSU);

    std::cout << "二值化：全局大津法（反相），阈值 = "
        << otsuValue << std::endl;

    // 3. 闭运算：先膨胀后腐蚀，填补小孔
    Mat kernel = getStructuringElement(MORPH_RECT, Size(3, 3));
    morphologyEx(binaryMat, binaryMat, MORPH_CLOSE, kernel);

    std::cout << "前景像素 = " << countNonZero(binaryMat) << std::endl;

    // 4. 连通域标记
    // labelCount包含背景，背景编号为0
    Mat labelMat;
    Mat stats;
    Mat centroids;

    int labelCount = connectedComponentsWithStats(
        binaryMat, labelMat, stats, centroids, 4, CV_32S);

    int originalLabelCount = labelCount - 1;

    // 5. 根据面积和外接矩形尺寸筛选回形针
    int noiseArea = 200;   // 小于该面积的连通域视为噪声
    int maxClipSize = 300; // 宽或高超过该像素值，视为其他物体

    std::vector<int> newIds(labelCount, 0);
    int objectCount = 0;

    for (int id = 1; id < labelCount; id++)
    {
        int w = stats.at<int>(id, CC_STAT_WIDTH);
        int h = stats.at<int>(id, CC_STAT_HEIGHT);
        int area = stats.at<int>(id, CC_STAT_AREA);

        bool tooLarge = w > maxClipSize || h > maxClipSize;

        // 只保留面积达标，并且宽、高都没有超过限制的区域
        if (area >= noiseArea && !tooLarge)
        {
            objectCount++;
            newIds[id] = objectCount;
        }
    }

    // 被过滤的区域设为背景0，保留的区域重新连续编号
    for (int j = 0; j < height; j++)
    {
        for (int i = 0; i < width; i++)
        {
            int id = labelMat.at<int>(j, i);
            if (id > 0)
            {
                labelMat.at<int>(j, i) = newIds[id];
            }
        }
    }

    std::cout << "连通域数量（不含背景）= " << objectCount << std::endl;

    std::cout << "\n========== 回形针数量 = " << objectCount
        << " ==========\n" << std::endl;

    int ignoredCount = originalLabelCount - objectCount;
    if (ignoredCount > 0)
    {
        std::cout << "（已忽略 " << ignoredCount
            << " 个小面积噪声或尺寸过大的非回形针目标）" << std::endl;
    }

    // 6. 输出区域统计表
    std::cout << '\n'
        << std::setw(5) << "序号"
        << std::setw(7) << "x"
        << std::setw(7) << "y"
        << std::setw(7) << "w"
        << std::setw(7) << "h"
        << std::setw(10) << "面积"
        << std::setw(11) << "cx"
        << std::setw(11) << "cy" << '\n';

    for (int id = 1; id < labelCount; id++)
    {
        if (id != 0 && newIds[id] == 0)
            continue;

        int area = stats.at<int>(id, CC_STAT_AREA);
        if (area == 0)
            continue;

        int outputId = newIds[id];

        std::cout << std::setw(5) << outputId
            << std::setw(7) << stats.at<int>(id, CC_STAT_LEFT)
            << std::setw(7) << stats.at<int>(id, CC_STAT_TOP)
            << std::setw(7) << stats.at<int>(id, CC_STAT_WIDTH)
            << std::setw(7) << stats.at<int>(id, CC_STAT_HEIGHT)
            << std::setw(10) << area
            << std::setw(11) << std::fixed << std::setprecision(1)
            << centroids.at<double>(id, 0)
            << std::setw(11) << centroids.at<double>(id, 1) << '\n';
    }

    // 7. 输出质心坐标
    std::cout << "\n中心坐标（cx = Σx/面积，cy = Σy/面积）：\n"
        << std::setw(5) << "序号"
        << std::setw(11) << "cx"
        << std::setw(11) << "cy"
        << std::setw(14) << "取整" << '\n';

    for (int id = 1; id < labelCount; id++)
    {
        if (id != 0 && newIds[id] == 0)
            continue;

        if (stats.at<int>(id, CC_STAT_AREA) == 0)
            continue;

        int outputId = newIds[id];
        double cx = centroids.at<double>(id, 0);
        double cy = centroids.at<double>(id, 1);

        std::cout << std::setw(5) << outputId
            << std::setw(11) << std::fixed << std::setprecision(2) << cx
            << std::setw(11) << cy
            << "    (" << static_cast<int>(cx + 0.5) << ", "
            << static_cast<int>(cy + 0.5) << ")" << '\n';
    }

    // 8. 给不同的连通域设置不同颜色，背景保持黑色
    const Scalar palette[] = {
        Scalar(0, 0, 255),
        Scalar(0, 165, 255),
        Scalar(0, 255, 255),
        Scalar(0, 255, 0),
        Scalar(255, 255, 0),
        Scalar(128, 255, 255),
        Scalar(255, 0, 0),
        Scalar(255, 0, 255),
        Scalar(128, 0, 255),
        Scalar(255, 128, 0),
        Scalar(128, 255, 0),
        Scalar(0, 128, 255),
        Scalar(203, 192, 255),
        Scalar(60, 20, 220),
        Scalar(40, 200, 120),
        Scalar(255, 0, 128),
        Scalar(128, 0, 0),
        Scalar(0, 0, 128)
    };

    int paletteSize = sizeof(palette) / sizeof(palette[0]);
    Mat labelView = Mat::zeros(height, width, CV_8UC3);

    for (int j = 0; j < height; j++)
    {
        for (int i = 0; i < width; i++)
        {
            int id = labelMat.at<int>(j, i);
            if (id > 0)
            {
                Scalar color = palette[id % paletteSize];
                labelView.at<Vec3b>(j, i)[0] = static_cast<uchar>(color[0]);
                labelView.at<Vec3b>(j, i)[1] = static_cast<uchar>(color[1]);
                labelView.at<Vec3b>(j, i)[2] = static_cast<uchar>(color[2]);
            }
        }
    }

    // 9. 绘制外接矩形、质心和编号
    Mat binaryView;
    Mat grayView;
    cvtColor(binaryMat, binaryView, COLOR_GRAY2BGR);
    cvtColor(srcMat, grayView, COLOR_GRAY2BGR);

    for (int id = 1; id < labelCount; id++)
    {
        if (newIds[id] == 0)
            continue;

        Rect box(
            stats.at<int>(id, CC_STAT_LEFT),
            stats.at<int>(id, CC_STAT_TOP),
            stats.at<int>(id, CC_STAT_WIDTH),
            stats.at<int>(id, CC_STAT_HEIGHT));

        Point center(
            static_cast<int>(centroids.at<double>(id, 0)),
            static_cast<int>(centroids.at<double>(id, 1)));

        Point textPos(box.x, std::max(box.y - 4, 12));
        std::string text = std::to_string(newIds[id]);

        rectangle(labelView, box, Scalar(0, 0, 255), 1);
        circle(labelView, center, 3, Scalar(255, 255, 255), -1);
        putText(labelView, text, textPos, FONT_HERSHEY_SIMPLEX,
            0.45, Scalar(0, 255, 0), 1);

        rectangle(grayView, box, Scalar(0, 0, 255), 1);
        circle(grayView, center, 3, Scalar(255, 255, 255), -1);
        putText(grayView, text, textPos, FONT_HERSHEY_SIMPLEX,
            0.45, Scalar(0, 255, 0), 1);
    }

    // 10. 保存并显示结果
    imwrite("out/binary.png", binaryView);
    imwrite("out/labels_color.png", labelView);
    imwrite("out/result.png", grayView);

    std::cout << "\n结果图已保存："
        << "out/binary.png、"
        << "out/labels_color.png、"
        << "out/result.png" << std::endl;

    imshow("1 binary", binaryView);
    imshow("2 labels color", labelView);
    imshow("3 gray + bbox", grayView);
    waitKey(0);

    return 0;
}