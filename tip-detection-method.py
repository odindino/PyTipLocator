# import cv2
# import numpy as np
# import matplotlib.pyplot as plt


# def detect_stm_tip(image_path):
#     # 讀取圖像
#     image = cv2.imread(image_path)
#     gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

#     # 應用自適應閾值處理
#     thresh = cv2.adaptiveThreshold(
#         gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 25, 2)

#     # 應用中值濾波來減少噪聲
#     median = cv2.medianBlur(thresh, 5)

#     # 應用膨脹操作來連接針的輪廓
#     kernel = np.ones((7, 7), np.uint8)
#     dilated = cv2.dilate(median, kernel, iterations=3)

#     # 找到輪廓
#     contours, _ = cv2.findContours(
#         dilated, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

#     # 過濾和選擇最合適的輪廓
#     suitable_contours = []
#     for cnt in contours:
#         area = cv2.contourArea(cnt)
#         if area > 500:  # 增加面積閾值以過濾小輪廓
#             x, y, w, h = cv2.boundingRect(cnt)
#             aspect_ratio = h / w
#             if aspect_ratio > 1.2:  # 稍微放寬長寬比例條件
#                 suitable_contours.append(cnt)

#     if suitable_contours:
#         # 選擇面積最大的合適輪廓
#         largest_contour = max(suitable_contours, key=cv2.contourArea)

#         # 獲取邊界框
#         x, y, w, h = cv2.boundingRect(largest_contour)

#         # 在原圖上繪製邊界框
#         cv2.rectangle(image, (x, y), (x+w, y+h), (0, 255, 0), 2)

#         # 顯示結果
#         plt.figure(figsize=(20, 5))
#         plt.subplot(151), plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
#         plt.title('Detected STM Tip'), plt.axis('off')
#         plt.subplot(152), plt.imshow(gray, cmap='gray')
#         plt.title('Grayscale'), plt.axis('off')
#         plt.subplot(153), plt.imshow(thresh, cmap='gray')
#         plt.title('Adaptive Threshold'), plt.axis('off')
#         plt.subplot(154), plt.imshow(median, cmap='gray')
#         plt.title('Median Blur'), plt.axis('off')
#         plt.subplot(155), plt.imshow(dilated, cmap='gray')
#         plt.title('Dilated'), plt.axis('off')
#         plt.show()

#         return x, y, w, h
#     else:
#         print("No suitable contour found")
#         return None


# # 測試函數
# image_path = 'tip detection_ex2.png'  # 請替換為您的圖像路徑
# bbox = detect_stm_tip(image_path)
# if bbox:
#     print(
#         f"Bounding box: x={bbox[0]}, y={bbox[1]}, width={bbox[2]}, height={bbox[3]}")
# import cv2
# import numpy as np
# import matplotlib.pyplot as plt


# def draw_contours(image, contours):
#     contour_image = image.copy()
#     cv2.drawContours(contour_image, contours, -1, (0, 255, 0), 2)
#     return contour_image


# def detect_stm_tip(image_path):
#     # 讀取圖像
#     image = cv2.imread(image_path)
#     gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

#     # 應用自適應閾值處理
#     thresh = cv2.adaptiveThreshold(
#         gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 25, 2)

#     # 找到閾值處理後的輪廓
#     thresh_contours, _ = cv2.findContours(
#         thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
#     thresh_contour_image = draw_contours(image, thresh_contours)

#     # 應用中值濾波來減少噪聲
#     median = cv2.medianBlur(thresh, 5)

#     # 找到中值濾波後的輪廓
#     median_contours, _ = cv2.findContours(
#         median, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
#     median_contour_image = draw_contours(image, median_contours)

#     # 應用膨脹操作來連接針的輪廓
#     kernel = np.ones((7, 7), np.uint8)
#     dilated = cv2.dilate(median, kernel, iterations=3)

#     # 找到輪廓
#     contours, _ = cv2.findContours(
#         dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

#     # 創建輪廓圖像
#     contour_image = draw_contours(image, contours)

#     # 過濾和選擇最合適的輪廓
#     suitable_contours = []
#     for cnt in contours:
#         area = cv2.contourArea(cnt)
#         if area > 500:  # 增加面積閾值以過濾小輪廓
#             x, y, w, h = cv2.boundingRect(cnt)
#             aspect_ratio = h / w
#             if aspect_ratio > 1.2:  # 稍微放寬長寬比例條件
#                 suitable_contours.append(cnt)

#     # 創建合適輪廓圖像
#     suitable_contour_image = draw_contours(image, suitable_contours)

#     if suitable_contours:
#         # 選擇面積最大的合適輪廓
#         largest_contour = max(suitable_contours, key=cv2.contourArea)

#         # 獲取邊界框
#         x, y, w, h = cv2.boundingRect(largest_contour)

#         # 在原圖上繪製邊界框
#         result_image = image.copy()
#         cv2.rectangle(result_image, (x, y), (x+w, y+h), (0, 255, 0), 2)

#         # 顯示結果
#         plt.figure(figsize=(20, 15))
#         plt.subplot(331), plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
#         plt.title('Original Image'), plt.axis('off')
#         plt.subplot(332), plt.imshow(gray, cmap='gray')
#         plt.title('Grayscale'), plt.axis('off')
#         plt.subplot(333), plt.imshow(thresh, cmap='gray')
#         plt.title('Adaptive Threshold'), plt.axis('off')
#         plt.subplot(334), plt.imshow(cv2.cvtColor(
#             thresh_contour_image, cv2.COLOR_BGR2RGB))
#         plt.title('Threshold Contours'), plt.axis('off')
#         plt.subplot(335), plt.imshow(median, cmap='gray')
#         plt.title('Median Blur'), plt.axis('off')
#         plt.subplot(336), plt.imshow(cv2.cvtColor(
#             median_contour_image, cv2.COLOR_BGR2RGB))
#         plt.title('Median Blur Contours'), plt.axis('off')
#         plt.subplot(337), plt.imshow(dilated, cmap='gray')
#         plt.title('Dilated'), plt.axis('off')
#         plt.subplot(338), plt.imshow(
#             cv2.cvtColor(contour_image, cv2.COLOR_BGR2RGB))
#         plt.title('Dilated Contours'), plt.axis('off')
#         plt.subplot(339), plt.imshow(
#             cv2.cvtColor(result_image, cv2.COLOR_BGR2RGB))
#         plt.title('Final Result'), plt.axis('off')
#         plt.tight_layout()
#         plt.show()

#         return x, y, w, h
#     else:
#         print("No suitable contour found")
#         return None


# # 測試函數
# image_path = 'tip detection_ex2.png'  # 請替換為您的圖像路徑
# bbox = detect_stm_tip(image_path)
# if bbox:
#     print(
# f"Bounding box: x={bbox[0]}, y={bbox[1]}, width={bbox[2]}, height={bbox[3]}")

# import cv2
# import numpy as np
# import matplotlib.pyplot as plt


# def draw_contours(image, contours):
#     contour_image = image.copy()
#     cv2.drawContours(contour_image, contours, -1, (0, 255, 0), 2)
#     return contour_image


# def detect_stm_tip(image_path):
#     # 讀取圖像
#     image = cv2.imread(image_path)
#     gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

#     # 應用自適應閾值處理
#     thresh = cv2.adaptiveThreshold(
#         gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 25, 2)

#     # 應用中值濾波來減少噪聲
#     median = cv2.medianBlur(thresh, 5)

#     # 應用膨脹操作來連接針的輪廓
#     kernel = np.ones((7, 7), np.uint8)
#     dilated = cv2.dilate(median, kernel, iterations=3)

#     # 找到輪廓
#     contours, _ = cv2.findContours(
#         dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

#     # 創建輪廓圖像
#     contour_image = draw_contours(image.copy(), contours)

#     if contours:
#         # 選擇面積最大的輪廓
#         largest_contour = max(contours, key=cv2.contourArea)

#         # 獲取邊界框
#         x, y, w, h = cv2.boundingRect(largest_contour)

#         # 在原圖上繪製邊界框
#         result_image = image.copy()
#         cv2.rectangle(result_image, (x, y), (x+w, y+h), (0, 255, 0), 2)

#         # 顯示結果
#         plt.figure(figsize=(20, 15))
#         plt.subplot(331), plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
#         plt.title('Original Image'), plt.axis('off')
#         plt.subplot(332), plt.imshow(gray, cmap='gray')
#         plt.title('Grayscale'), plt.axis('off')
#         plt.subplot(333), plt.imshow(thresh, cmap='gray')
#         plt.title('Adaptive Threshold'), plt.axis('off')
#         plt.subplot(334), plt.imshow(median, cmap='gray')
#         plt.title('Median Blur'), plt.axis('off')
#         plt.subplot(335), plt.imshow(dilated, cmap='gray')
#         plt.title('Dilated'), plt.axis('off')
#         plt.subplot(336), plt.imshow(
#             cv2.cvtColor(contour_image, cv2.COLOR_BGR2RGB))
#         plt.title('All Contours'), plt.axis('off')
#         plt.subplot(337), plt.imshow(
#             cv2.cvtColor(result_image, cv2.COLOR_BGR2RGB))
#         plt.title('Final Result'), plt.axis('off')
#         plt.tight_layout()
#         plt.show()

#         return x, y, w, h
#     else:
#         print("No contour found")
#         return None


# # 測試函數
# image_path = 'tip detection_ex6.png'  # 請替換為您的圖像路徑
# bbox = detect_stm_tip(image_path)
# if bbox:
#     print(
#         f"Bounding box: x={bbox[0]}, y={bbox[1]}, width={bbox[2]}, height={bbox[3]}")
import cv2
import numpy as np
import matplotlib.pyplot as plt


def draw_contours(image, contours, color=(0, 255, 0)):
    contour_image = image.copy()
    cv2.drawContours(contour_image, contours, -1, color, 2)
    return contour_image


def detect_stm_tip(image_path):
    # 讀取圖像
    image = cv2.imread(image_path)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # 應用自適應閾值處理
    thresh = cv2.adaptiveThreshold(
        gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 25, 2)

    # 應用中值濾波來減少噪聲
    median = cv2.medianBlur(thresh, 5)

    # 應用膨脹操作來連接針的輪廓
    kernel = np.ones((11, 3), np.uint8)
    dilated = cv2.dilate(median, kernel, iterations=3)

    # 找到輪廓
    contours, _ = cv2.findContours(
        dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # 按面積排序輪廓
    sorted_contours = sorted(contours, key=cv2.contourArea, reverse=True)

    if len(sorted_contours) >= 2:
        # 獲取最大的兩個輪廓
        largest_contour = sorted_contours[0]
        second_largest_contour = sorted_contours[1]

        # 計算兩個輪廓的邊界框面積
        x1, y1, w1, h1 = cv2.boundingRect(largest_contour)
        x2, y2, w2, h2 = cv2.boundingRect(second_largest_contour)
        area1 = w1 * h1
        area2 = w2 * h2

        # 選擇面積較大的邊界框作為針的輪廓
        if area1 > area2:
            x, y, w, h = x1, y1, w1, h1
            selected_contour = largest_contour
        else:
            x, y, w, h = x2, y2, w2, h2
            selected_contour = second_largest_contour

        # 在原圖上繪製選中的邊界框
        result_image = image.copy()
        cv2.rectangle(result_image, (x, y), (x+w, y+h), (0, 255, 0), 2)

        # 顯示結果
        plt.figure(figsize=(20, 15))
        plt.subplot(331), plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        plt.title('Original Image'), plt.axis('off')
        plt.subplot(332), plt.imshow(gray, cmap='gray')
        plt.title('Grayscale'), plt.axis('off')
        plt.subplot(333), plt.imshow(thresh, cmap='gray')
        plt.title('Adaptive Threshold'), plt.axis('off')
        plt.subplot(334), plt.imshow(median, cmap='gray')
        plt.title('Median Blur'), plt.axis('off')
        plt.subplot(335), plt.imshow(dilated, cmap='gray')
        plt.title('Dilated'), plt.axis('off')
        plt.subplot(336), plt.imshow(cv2.cvtColor(draw_contours(
            image, [largest_contour], (255, 0, 0)), cv2.COLOR_BGR2RGB))
        plt.title('Largest Contour'), plt.axis('off')
        plt.subplot(337), plt.imshow(cv2.cvtColor(draw_contours(
            image, [second_largest_contour], (0, 0, 255)), cv2.COLOR_BGR2RGB))
        plt.title('Second Largest Contour'), plt.axis('off')
        plt.subplot(338), plt.imshow(
            cv2.cvtColor(result_image, cv2.COLOR_BGR2RGB))
        plt.title('Final Result'), plt.axis('off')
        plt.tight_layout()
        plt.show()

        return x, y, w, h
    else:
        print("Not enough contours found")
        return None


# 測試函數
image_path = 'tip detection_ex3.png'  # 請替換為您的圖像路徑
bbox = detect_stm_tip(image_path)
if bbox:
    print(
        f"Bounding box: x={bbox[0]}, y={bbox[1]}, width={bbox[2]}, height={bbox[3]}")
