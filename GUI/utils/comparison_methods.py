#!/usr/bin/env python3
# Comparison algorithms for analyzing PLY model differences
# Modified to only include dental tartar detection

from PyQt5.QtGui import QImage, QColor


class ComparisonMethods:
    """Collection of static methods for comparing images - focused on dental tartar detection"""

    @staticmethod
    def detect_dental_tartar(img1, img2, width, height, threshold):
        """Specialized method to detect tartar in dental models using before/after tablet comparison.

        img1 = Before tablet (no pink)
        img2 = After tablet (pink areas indicate tartar)
        """
        diff_img = QImage(width, height, QImage.Format_RGB32)
        tartar_count = 0
        total_pixels = width * height

        # Define pink color range for tartar in model 2 (after tablet)
        pink_min_r = 180  # Minimum red value for pink
        pink_min_g = 80  # Minimum green value for pink
        pink_min_b = 80  # Minimum blue value for pink

        pink_max_r = 255  # Maximum red value for pink
        pink_max_g = 180  # Maximum green value for pink
        pink_max_b = 180  # Maximum blue value for pink

        for y in range(height):
            for x in range(width):
                # Skip black background pixels
                color1 = QColor(img1.pixel(x, y))
                color2 = QColor(img2.pixel(x, y))

                if color1.red() < 10 and color1.green() < 10 and color1.blue() < 10:
                    diff_img.setPixel(x, y, QColor(0, 0, 0).rgb())  # Black for background
                    continue

                # Extract RGB values
                r1, g1, b1 = color1.red(), color1.green(), color1.blue()
                r2, g2, b2 = color2.red(), color2.green(), color2.blue()

                # Check if pixel became pink in after-tablet model (model 2)
                is_pink = (
                        pink_min_r <= r2 <= pink_max_r and
                        pink_min_g <= g2 <= pink_max_g and
                        pink_min_b <= b2 <= pink_max_b and
                        # Red channel significantly higher than blue
                        r2 > (b2 + threshold)
                )

                # Additional check that it wasn't already pink in model 1
                was_not_pink = not (
                        pink_min_r <= r1 <= pink_max_r and
                        pink_min_g <= g1 <= pink_max_g and
                        pink_min_b <= b1 <= pink_max_b and
                        r1 > (b1 + threshold)
                )

                if is_pink and was_not_pink:
                    # Tartar area: became pink in model 2
                    diff_img.setPixel(x, y, QColor(255, 0, 0).rgb())  # Red for tartar
                    tartar_count += 1
                elif abs(r1 - r2) + abs(g1 - g2) + abs(b1 - b2) > threshold * 3:
                    # Other significant color difference
                    diff_img.setPixel(x, y, QColor(0, 0, 255).rgb())  # Blue for other differences
                else:
                    # No significant difference
                    diff_img.setPixel(x, y, QColor(0, 255, 0).rgb())  # Green for matching areas

        return diff_img, tartar_count, total_pixels