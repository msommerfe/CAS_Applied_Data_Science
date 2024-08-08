import os
import random
from PIL import Image, ImageDraw, ImageFont, ImageFilter


# Erstelle ein Verzeichnis, um die Bilder zu speichern
output_dir = '/mnt/c/dev/git/CAS_Applied_Data_Science/CAS_Final_Project/generated_images'

if not os.path.exists(output_dir):
    os.makedirs(output_dir)


# Funktion zum Generieren einer zufälligen 12-stelligen Zahl
def generate_random_number():
    return ''.join(random.choices('0123456789', k=12))


# Funktion zum Erstellen eines zufälligen Hintergrundes
def generate_random_background(width, height):
    img = Image.new('RGB', (width, height),
                    color=(random.randint(200, 255), random.randint(200, 255), random.randint(200, 255)))
    draw = ImageDraw.Draw(img)
    # Optional: Muster hinzufügen
    for _ in range(random.randint(5, 15)):
        x1 = random.randint(0, width)
        y1 = random.randint(0, height)
        x2 = random.randint(0, width)
        y2 = random.randint(0, height)
        draw.line((x1, y1, x2, y2), fill=(random.randint(0, 255), random.randint(0, 255), random.randint(0, 255)),
                  width=1)
    return img


# Funktion zum Laden einer Schriftart
def load_font(font_size):
    try:
        return ImageFont.truetype("/mnt/c/dev/git/CAS_Applied_Data_Science/CAS_Final_Project/fonts/arial_narrow_bold.ttf", font_size)
    except IOError:
        return ImageFont.load_default()


# Funktion zum Berechnen der Begrenzungsbox aller Ziffern
def calculate_bounding_box(draw, text, font, start_x, start_y):
    min_x = start_x
    min_y = start_y
    max_x = start_x
    max_y = start_y
    current_x = start_x
    current_y = start_y

    for char in text:
        bbox = draw.textbbox((current_x, current_y), char, font=font)
        min_x = min(min_x, bbox[0])
        min_y = min(min_y, bbox[1])
        max_x = max(max_x, bbox[2])
        max_y = max(max_y, bbox[3])

        char_width = bbox[2] - bbox[0]
        current_x += char_width + random.randint(2, 10)

    return (min_x, min_y, max_x, max_y)

for i in range(50):
    # Zufällige Bildgrösse
    img_width = random.randint(800, 1000)
    img_height = random.randint(500, 1000)

    # Neues Bild mit zufälligem Hintergrund erstellen
    img = generate_random_background(img_width, img_height)
    d = ImageDraw.Draw(img)

    # Zufällige 12-stellige Zahl generieren
    random_number = generate_random_number()

    # Zufällige Startposition für den Text
    initial_x = random.randint(10, 50)  # Zufällige Startposition mit etwas Rand
    initial_y = random.randint(10, 50)  # Zufällige Startposition mit etwas Rand

    current_x = initial_x
    current_y = initial_y

    min_x = current_x
    min_y = current_y
    max_x = current_x
    max_y = current_y

    # Zufällige Grautonfarbe für die Ziffer
    gray_value = random.randint(0, 150)
    text_color = (gray_value, gray_value, gray_value)


    # Zufällige Schriftgrösse
    font_size = random.randint(8, 128)
    font = load_font(font_size)
    oldXoffset = 0
    for char in random_number:

        # Zufällige Verschiebung für jede Ziffer in y-Richtung
        offset_x = random.randint(0, 20)
        offset_y = random.randint(-1, 1)

        # Zeichne die Ziffer mit zufälliger y-Verschiebung und Schriftgrösse
        d.text((current_x, current_y + offset_y), char, fill=text_color, font=font)

        # Berechne die Begrenzungsbox der aktuellen Ziffer
        bbox = d.textbbox((current_x + offset_x, current_y + offset_y), char, font=font)
        min_x = min(min_x, bbox[0])
        min_y = min(min_y, bbox[1])
        max_x = max(max_x, bbox[2])
        max_y = max(max_y, bbox[3])

        # Aktualisiere die x-Position für die nächste Ziffer
        char_width = d.textbbox((0, 0), char, font=font)[2] - d.textbbox((0, 0), char, font=font)[0]
        current_x += char_width + random.randint(2, 10)  # Zufälliger Abstand zwischen den Ziffern

        # Überprüfe, ob die nächste Ziffer noch ins Bild passt
        if current_x + char_width > img_width:
            print("nuber not fully displayed in image")
            break

    # Zufällige Unschärfe anwenden
    blur_radius = random.uniform(0, 2)  # Radius zwischen 0 und 2
    img = img.filter(ImageFilter.GaussianBlur(blur_radius))

    # Füge einen kleinen Rand hinzu
    padding = 10
    bbox = (
    max(0, min_x - padding), max(0, min_y - padding), min(img_width, max_x + padding), min(img_height, max_y + padding))

    # Zuschneiden des Bildes
    img_cropped = img.crop(bbox)

    # Zuschneiden des Bildes
    img = img.crop(bbox)

    # Bild speichern
    img.save(os.path.join(output_dir, f'image_{i:04d}.png'))

print("Bilder wurden erfolgreich generiert und gespeichert.")