import cv2
import numpy as np

from CFG import ROAD_IMAGE_PATH


def get_road_contour(image_path):
    """
    Verilen bir yol resminde yolun ana hatlarını çıkarır.
    """
    # Resmi yükle ve gri tona dönüştür
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise FileNotFoundError(f"Resim bulunamadı: {image_path}")

    # Eşikleme (thresholding) işlemi
    _, binary = cv2.threshold(image, 127, 255, cv2.THRESH_BINARY_INV)

    # Yolun konturunu çıkar
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not contours:
        raise ValueError("Yol konturu bulunamadı.")

    # En büyük konturu al (yol olarak kabul edilir)
    road_contour = max(contours, key=cv2.contourArea)
    return road_contour


def generate_points_on_road(contour, num_points):
    """
    Yol konturunda belirli sayıda eşit aralıklı noktalar oluşturur.
    """
    # Konturdaki toplam uzunluğu hesapla
    total_length = cv2.arcLength(contour, True)
    step = total_length / num_points

    # Kontur boyunca eşit aralıklı noktaları hesapla
    road_points = []
    current_length = 0
    for i in range(len(contour)):
        # Bir sonraki kontur noktasına geçerken mesafeyi artır
        next_length = cv2.arcLength(contour[:i + 1], False)
        if next_length >= current_length:
            road_points.append(tuple(contour[i][0]))  # Kontur noktasını kaydet
            current_length += step
        if len(road_points) >= num_points:  # İstenilen nokta sayısına ulaşıldıysa dur
            break

    return road_points


def save_road_points(road_points, filepath):
    """
    Oluşturulan yol noktalarını bir dosyaya kaydeder.
    """
    with open(filepath, "w") as file:
        for point in road_points:
            file.write(f"{point[0]},{point[1]}\n")
    print(f"{len(road_points)} nokta '{filepath}' dosyasına kaydedildi.")


if __name__ == "__main__":
    # Resim yolu ve çıkış dosyası
    image_path = f'{ROAD_IMAGE_PATH}' # Yol resmi
    output_file = "road_points.txt"

    # Yol konturunu al
    try:
        road_contour = get_road_contour(image_path)

        # Nokta sayısı
        num_points = 250

        # Yol üzerinde eşit aralıklı noktalar oluştur
        road_points = generate_points_on_road(road_contour, num_points)

        # Yol noktalarını kaydet
        save_road_points(road_points, output_file)
    except Exception as e:
        print(f"Hata: {e}")
