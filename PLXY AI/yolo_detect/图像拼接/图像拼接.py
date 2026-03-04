import sys
import cv2

if __name__ == "__main__":
    img1 = cv2.imread(r'A4-1-6-IN12-20X (2).jpg')  # 图片绝对路径
    img2 = cv2.imread(r'A4-1-6-IN12-20X.jpg')

    # 创建 Stitcher 对象进行全景拼接
    stitcher = cv2.Stitcher.create(cv2.Stitcher_PANORAMA)

    (status, pano) = stitcher.stitch((img1, img2))

    # 判断拼接是否成功
    if status != cv2.Stitcher_OK:
        print("不能拼接图片, error code = %d" % status)
        sys.exit(-1)

    print("拼接成功.")

    # 显示拼接图像
    cv2.imshow('pano', pano)

    # 保存拼接结果为图片文件
    output_path = r'A4-1-6-IN12-20Xping.jpg'  # 保存路径和文件名（可以修改为 .png 等其他格式）
    save_success = cv2.imwrite(output_path, pano)

    if save_success:
        print(f"拼接结果已保存到：{output_path}")
    else:
        print(f"无法保存拼接结果到：{output_path}")

    cv2.waitKey(0)
    cv2.destroyAllWindows()
